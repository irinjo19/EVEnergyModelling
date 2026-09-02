"""
VED EV Energy Modeling Pipeline
================================

Compares four approaches for predicting EV battery energy consumption
from real-world telemetry in the Vehicle Energy Dataset (VED):

  1. Physics-only     - force-balance vehicle dynamics + fixed drivetrain efficiency
  2. Data-NN          - pure data-driven feedforward network (no physics)
  3. PINN             - same network, with a physics-consistency loss term
  4. Residual NN      - network predicts the residual (P_bat - P_physics),
                        physics estimate added back afterward

Evaluation design:
  - Train / Validation / Test-SV: Vehicles 10 and 455, split by whole trip
  - Test-NV: Vehicle 541, held out completely (never seen in training or tuning)

Usage:
    python ved_ev_energy_pipeline.py
    python ved_ev_energy_pipeline.py --output-dir results --no-plots

Requirements: see requirements.txt (pandas, numpy, torch, requests, py7zr,
openpyxl, matplotlib). Internet access is required on first run to download
VED from GitHub (~170 MB); the extracted files are cached locally afterward.
"""

import argparse
import os
import random
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn

try:
    import py7zr
except ImportError as e:
    raise ImportError(
        "py7zr is required to extract the VED archive. Install with: pip install py7zr"
    ) from e

try:
    import matplotlib
    matplotlib.use("Agg")  # safe for headless/local runs; figures are saved to disk
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================
# Configuration
# ============================================================

SEED = 42

# Vehicle/physics constants (2013 Nissan Leaf, the only pure-EV model in VED)
MASS_KG = 3500 * 0.453592
CD = 0.28
FRONTAL_AREA_M2 = 2.27
CRR = 0.01
RHO_AIR = 1.225
G = 9.81

SMOOTH_WINDOW = 5      # rolling-average window for speed smoothing before differentiation
MAX_DT_S = 5.0         # timestamp gaps above this are treated as logging breaks, not motion
ACCEL_LIMIT = 6.0      # m/s^2, physically realistic bound for a passenger EV
MIN_TRIP_SAMPLES = 100  # minimum rows for a trip to be included in trip-level evaluation
MIN_SPEED_MS = 2.0      # rows below this speed are excluded (idle/near-stationary)
ETA_FIXED = 0.70        # drivetrain efficiency, independently validated via trip-level
                         # energy regression (see paper Section III); NOT learned by the PINN

PINN_LAMBDAS = [0.1, 1.0, 5.0, 20.0]
FEATURES = ["speed_ms_smooth", "accel_ms2", "P_aux_W", "OAT[DegC]"]
RESIDUAL_FEATURES = FEATURES + ["P_physics_W"]

VED_STATIC_URL = "https://github.com/gsoh/VED/raw/master/Data/VED_Static_Data_PHEV%26EV.xlsx"
VED_DYNAMIC_URLS = {
    "Part1": "https://raw.githubusercontent.com/gsoh/VED/master/Data/VED_DynamicData_Part1.7z",
    "Part2": "https://raw.githubusercontent.com/gsoh/VED/master/Data/VED_DynamicData_Part2.7z",
}

COLS_TO_DROP = [
    "MAF[g/sec]", "Engine RPM[RPM]", "Absolute Load[%]", "Fuel Rate[L/hr]",
    "Air Conditioning Power[kW]",
    "Short Term Fuel Trim Bank 1[%]", "Short Term Fuel Trim Bank 2[%]",
    "Long Term Fuel Trim Bank 1[%]", "Long Term Fuel Trim Bank 2[%]",
]


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Step 1 - Download VED and retain the three pure-EV vehicles
# ============================================================

def download_ved(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    extract_dir = os.path.join(data_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    print("Fetching static vehicle spec file...")
    static_ev = pd.read_excel(BytesIO(requests.get(VED_STATIC_URL).content))
    ev_ids = static_ev.loc[static_ev["EngineType"].eq("EV"), "VehId"].tolist()
    print("Pure-EV VehIds:", ev_ids)
    assert set(ev_ids) == {10, 455, 541}, (
        f"Expected VehIds {{10, 455, 541}}, got {set(ev_ids)}. "
        "VED's static file may have changed upstream."
    )

    for name, url in VED_DYNAMIC_URLS.items():
        archive_path = os.path.join(data_dir, f"VED_DynamicData_{name}.7z")
        if not os.path.exists(archive_path):
            print(f"Downloading {name} (~80-90 MB)...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(archive_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
        print(f"Extracting {name}...")
        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            archive.extractall(path=extract_dir)

    csv_files = sorted(
        os.path.join(extract_dir, f)
        for f in os.listdir(extract_dir)
        if f.endswith(".csv")
    )
    if not csv_files:
        raise RuntimeError(f"No CSV files found after extraction in {extract_dir}")

    ev_chunks = []
    for path in csv_files:
        chunk = pd.read_csv(path)
        chunk = chunk[chunk["VehId"].isin(ev_ids)].copy()
        if not chunk.empty:
            ev_chunks.append(chunk)

    ev_df = pd.concat(ev_chunks, ignore_index=True)
    print(f"EV rows: {len(ev_df)}")
    print("Rows by vehicle:")
    print(ev_df["VehId"].value_counts().sort_index().to_string())
    return ev_df


# ============================================================
# Step 2 - Leakage-safe Train / Val / Test-SV / Test-NV splits
# ============================================================

def make_splits(ev_df, seed=SEED):
    test_nv_df = ev_df[ev_df["VehId"] == 541].copy()
    trainval_df = ev_df[ev_df["VehId"].isin([10, 455])].copy()
    trainval_df["veh_trip_key"] = list(zip(trainval_df["VehId"], trainval_df["Trip"]))

    rng = np.random.default_rng(seed)
    train_keys, val_keys, test_sv_keys = [], [], []

    for _veh_id, group in trainval_df.groupby("VehId"):
        keys = np.array(group["veh_trip_key"].unique(), dtype=object)
        rng.shuffle(keys)
        n = len(keys)
        n_val = int(n * 0.15)
        n_test = int(n * 0.15)
        val_keys.extend(keys[:n_val])
        test_sv_keys.extend(keys[n_val:n_val + n_test])
        train_keys.extend(keys[n_val + n_test:])

    train_df = trainval_df[trainval_df["veh_trip_key"].isin(train_keys)].copy()
    val_df = trainval_df[trainval_df["veh_trip_key"].isin(val_keys)].copy()
    test_sv_df = trainval_df[trainval_df["veh_trip_key"].isin(test_sv_keys)].copy()

    assert not (set(train_keys) & set(val_keys))
    assert not (set(train_keys) & set(test_sv_keys))
    assert not (set(val_keys) & set(test_sv_keys))

    print(f"Train:   {len(train_df):,} rows")
    print(f"Val:     {len(val_df):,} rows")
    print(f"Test-SV: {len(test_sv_df):,} rows")
    print(f"Test-NV: {len(test_nv_df):,} rows")
    return train_df, val_df, test_sv_df, test_nv_df


# ============================================================
# Step 3 - Preprocessing (identical for all four splits)
# ============================================================

def preprocess_split(df):
    d = df.copy()
    d.drop(columns=[c for c in COLS_TO_DROP if c in d.columns], inplace=True)
    d.sort_values(["VehId", "Trip", "Timestamp(ms)"], inplace=True)
    d.reset_index(drop=True, inplace=True)

    # Measured battery power; sign convention: positive = battery discharge
    d["P_bat_W"] = d["HV Battery Voltage[V]"] * d["HV Battery Current[A]"]
    d["P_bat_out_W"] = -d["P_bat_W"]

    d["speed_ms"] = d["Vehicle Speed[km/h]"] / 3.6

    # Smooth speed BEFORE differentiating -- raw GPS-derived speed jitter,
    # when differentiated directly, produces physically impossible acceleration
    # spikes (observed up to +-48 m/s^2 without this step).
    d["speed_ms_smooth"] = (
        d.groupby(["VehId", "Trip"])["speed_ms"]
         .transform(lambda x: x.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean())
    )

    # Use the actual (irregular) timestamp spacing, not an assumed fixed rate
    d["dt_s"] = d.groupby(["VehId", "Trip"])["Timestamp(ms)"].diff() / 1000.0
    d["accel_ms2"] = d.groupby(["VehId", "Trip"])["speed_ms_smooth"].diff() / d["dt_s"]

    # Duplicate/reversed timestamps or large logging gaps are not continuous motion
    invalid_dt = (d["dt_s"] <= 0) | (d["dt_s"] > MAX_DT_S)
    d.loc[invalid_dt, "accel_ms2"] = np.nan
    d["accel_ms2"] = d["accel_ms2"].clip(-ACCEL_LIMIT, ACCEL_LIMIT)

    v = d["speed_ms_smooth"].to_numpy()
    a = d["accel_ms2"].to_numpy()
    F_roll = CRR * MASS_KG * G
    F_aero = 0.5 * RHO_AIR * CD * FRONTAL_AREA_M2 * v ** 2
    F_inertia = MASS_KG * a
    d["P_wheel_W"] = (F_roll + F_aero + F_inertia) * v

    d["P_aux_W"] = (
        d["Air Conditioning Power[Watts]"].fillna(0) + d["Heater Power[Watts]"].fillna(0)
    )
    return d


# ============================================================
# Step 4 - Shared evaluation utilities
# ============================================================

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    wape = np.sum(np.abs(err)) / np.sum(np.abs(y_true)) * 100
    return {"MAE_Wh": mae, "RMSE_Wh": rmse, "R2": r2, "WAPE_pct": wape}


def integrate_trip_energy(rows_df, pred_power):
    d = rows_df.copy()
    d["P_pred_W"] = np.asarray(pred_power)
    trip = (
        d.groupby(["VehId", "Trip"])
         .apply(
             lambda g: pd.Series({
                 "E_true_Wh": np.sum(g["P_bat_out_W"] * g["dt_s"]) / 3600,
                 "E_pred_Wh": np.sum(g["P_pred_W"] * g["dt_s"]) / 3600,
                 "n_samples": len(g),
             }),
             include_groups=False,
         )
         .reset_index()
    )
    return trip[trip["n_samples"] >= MIN_TRIP_SAMPLES].copy()


def evaluate_trip_table(trip_df):
    return compute_metrics(trip_df["E_true_Wh"], trip_df["E_pred_Wh"])


def valid_rows(df):
    required = ["speed_ms_smooth", "accel_ms2", "P_aux_W", "OAT[DegC]", "P_bat_out_W", "dt_s"]
    return df[(df["speed_ms_smooth"] > MIN_SPEED_MS)].dropna(subset=required).copy()


# ============================================================
# Step 5 - Physics-only baseline
# ============================================================

def physics_power(df, eta=ETA_FIXED):
    v = df["speed_ms_smooth"].to_numpy()
    a = df["accel_ms2"].to_numpy()
    aux = df["P_aux_W"].to_numpy()
    F_roll = CRR * MASS_KG * G
    F_aero = 0.5 * RHO_AIR * CD * FRONTAL_AREA_M2 * v ** 2
    F_inertia = MASS_KG * a
    P_wheel = (F_roll + F_aero + F_inertia) * v
    return P_wheel / eta + aux


# ============================================================
# Step 6 - Models
# ============================================================

class BaseNN(nn.Module):
    """Small feedforward network shared by the Data-NN, PINN, and Residual NN."""

    def __init__(self, n_inputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 32), nn.Tanh(),
            nn.Linear(32, 16), nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PINNFixedEta(BaseNN):
    """Same architecture as BaseNN, plus a fixed-eta physics-consistency term.

    eta is fixed (not learned) because an earlier experiment letting eta float
    jointly with the network weights caused it to drift toward an implausible
    value (~0.95) without stabilizing -- a parameter-identifiability failure.
    """

    def __init__(self, eta_fixed=ETA_FIXED):
        super().__init__(4)
        self.register_buffer("eta", torch.tensor(float(eta_fixed)))

    def physics_pred_normalized(self, X_raw, y_mean, y_std):
        v, a, aux = X_raw[:, 0], X_raw[:, 1], X_raw[:, 2]
        F_roll = CRR * MASS_KG * G
        F_aero = 0.5 * RHO_AIR * CD * FRONTAL_AREA_M2 * v ** 2
        F_inertia = MASS_KG * a
        P_wheel = (F_roll + F_aero + F_inertia) * v
        P_phys = P_wheel / self.eta + aux
        return (P_phys - y_mean.to(X_raw.device)) / y_std.to(X_raw.device)


def train_generic(model, X_train_n, y_train_n, X_val_n, y_val_n,
                   physics_term_fn=None, lambda_physics=0.0,
                   max_epochs=1000, patience=30, lr=0.005, seed=SEED):
    """Shared training loop for Data-NN, PINN, and Residual NN.

    physics_term_fn(model, X_train_n) -> normalized physics prediction tensor,
    used only when lambda_physics > 0 (i.e. for the PINN).
    """
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = np.inf
    best_state = None
    stale = 0

    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        pred_n = model(X_train_n)
        data_loss = torch.mean((pred_n - y_train_n) ** 2)

        if physics_term_fn is not None and lambda_physics > 0:
            physics_n = physics_term_fn(model, X_train_n)
            loss = data_loss + lambda_physics * torch.mean((pred_n - physics_n) ** 2)
        else:
            loss = data_loss

        loss.backward()
        opt.step()

        if (epoch + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                val_mse = torch.mean((model(X_val_n) - y_val_n) ** 2).item()
            if val_mse < best_val:
                best_val = val_mse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

    if best_state is None:
        raise RuntimeError("Model never saved a validation checkpoint.")
    model.load_state_dict(best_state)
    return model, best_val


def predict_power(model, Xn, y_mean, y_std):
    model.eval()
    with torch.no_grad():
        return (model(Xn) * y_std.to(Xn.device) + y_mean.to(Xn.device)).cpu().numpy()


# ============================================================
# Main pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="VED EV energy modeling pipeline")
    parser.add_argument("--data-dir", default="./ved_data",
                         help="Directory to download/cache VED data (default: ./ved_data)")
    parser.add_argument("--output-dir", default="./results",
                         help="Directory to write result CSVs and plots (default: ./results)")
    parser.add_argument("--no-plots", action="store_true",
                         help="Skip generating the physics-weight sensitivity plot")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_all_seeds(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- Data ----
    ev_df = download_ved(args.data_dir)
    train_df, val_df, test_sv_df, test_nv_df = make_splits(ev_df)

    train_df = preprocess_split(train_df)
    val_df = preprocess_split(val_df)
    test_sv_df = preprocess_split(test_sv_df)
    test_nv_df = preprocess_split(test_nv_df)

    # ---- Physics-only baseline ----
    splits = {
        "Train": valid_rows(train_df), "Val": valid_rows(val_df),
        "Test-SV": valid_rows(test_sv_df), "Test-NV": valid_rows(test_nv_df),
    }
    physics_metrics = {}
    for name, rows in splits.items():
        trip = integrate_trip_energy(rows, physics_power(rows))
        physics_metrics[name] = evaluate_trip_table(trip)

    # ---- Shared tensors for Data-NN / PINN ----
    def make_tensors(df):
        rows = valid_rows(df)
        X = torch.tensor(rows[FEATURES].to_numpy(), dtype=torch.float32)
        y = torch.tensor(rows["P_bat_out_W"].to_numpy(), dtype=torch.float32)
        return X, y, rows

    X_train_t, y_train_t, train_rows = make_tensors(train_df)
    X_val_t, y_val_t, val_rows = make_tensors(val_df)
    X_test_sv_t, y_test_sv_t, test_sv_rows = make_tensors(test_sv_df)
    X_test_nv_t, y_test_nv_t, test_nv_rows = make_tensors(test_nv_df)

    X_mean, X_std = X_train_t.mean(0), X_train_t.std(0).clamp_min(1e-8)
    y_mean, y_std = y_train_t.mean(), y_train_t.std().clamp_min(1e-8)

    def norm_X(X):
        return ((X - X_mean) / X_std).to(device)

    X_train_n, X_val_n = norm_X(X_train_t), norm_X(X_val_t)
    X_test_sv_n, X_test_nv_n = norm_X(X_test_sv_t), norm_X(X_test_nv_t)
    y_train_n = ((y_train_t - y_mean) / y_std).to(device)
    y_val_n = ((y_val_t - y_mean) / y_std).to(device)
    X_train_raw = X_train_t.to(device)

    def eval_model_all_splits(model, y_mean, y_std):
        return {
            "Train": evaluate_trip_table(integrate_trip_energy(
                train_rows, predict_power(model, X_train_n, y_mean, y_std))),
            "Val": evaluate_trip_table(integrate_trip_energy(
                val_rows, predict_power(model, X_val_n, y_mean, y_std))),
            "Test-SV": evaluate_trip_table(integrate_trip_energy(
                test_sv_rows, predict_power(model, X_test_sv_n, y_mean, y_std))),
            "Test-NV": evaluate_trip_table(integrate_trip_energy(
                test_nv_rows, predict_power(model, X_test_nv_n, y_mean, y_std))),
        }

    # ---- Data-NN ----
    print("\nTraining Data-NN...")
    data_model = BaseNN(4).to(device)
    data_model, _ = train_generic(data_model, X_train_n, y_train_n, X_val_n, y_val_n)
    data_metrics = eval_model_all_splits(data_model, y_mean, y_std)

    # ---- PINN lambda sweep ----
    pinn_results = {}
    for lam in PINN_LAMBDAS:
        print(f"Training PINN: lambda={lam}")
        model = PINNFixedEta().to(device)
        physics_fn = lambda m, Xn: m.physics_pred_normalized(X_train_raw, y_mean, y_std)
        model, _ = train_generic(
            model, X_train_n, y_train_n, X_val_n, y_val_n,
            physics_term_fn=physics_fn, lambda_physics=lam,
        )
        pinn_results[lam] = eval_model_all_splits(model, y_mean, y_std)

    best_lambda = min(PINN_LAMBDAS, key=lambda l: pinn_results[l]["Val"]["WAPE_pct"])
    print("Validation-selected PINN lambda:", best_lambda)

    lambda_summary = pd.DataFrame({
        "lambda": [0.0] + PINN_LAMBDAS,
        "Test-SV_WAPE": [data_metrics["Test-SV"]["WAPE_pct"]] +
                         [pinn_results[l]["Test-SV"]["WAPE_pct"] for l in PINN_LAMBDAS],
        "Test-NV_WAPE": [data_metrics["Test-NV"]["WAPE_pct"]] +
                         [pinn_results[l]["Test-NV"]["WAPE_pct"] for l in PINN_LAMBDAS],
    })

    # ---- True additive residual-learning NN ----
    print("\nTraining Residual NN...")

    def make_residual_tensors(df):
        rows = valid_rows(df).copy()
        rows["P_physics_W"] = physics_power(rows, eta=ETA_FIXED)
        rows["residual_true_W"] = rows["P_bat_out_W"] - rows["P_physics_W"]
        X = torch.tensor(rows[RESIDUAL_FEATURES].to_numpy(), dtype=torch.float32)
        y = torch.tensor(rows["residual_true_W"].to_numpy(), dtype=torch.float32)
        return X, y, rows

    Xtr_r, ytr_r, rows_tr_r = make_residual_tensors(train_df)
    Xva_r, yva_r, rows_va_r = make_residual_tensors(val_df)
    Xsv_r, ysv_r, rows_sv_r = make_residual_tensors(test_sv_df)
    Xnv_r, ynv_r, rows_nv_r = make_residual_tensors(test_nv_df)

    Xm_r, Xs_r = Xtr_r.mean(0), Xtr_r.std(0).clamp_min(1e-8)
    ym_r, ys_r = ytr_r.mean(), ytr_r.std().clamp_min(1e-8)

    Xtr_rn = ((Xtr_r - Xm_r) / Xs_r).to(device)
    Xva_rn = ((Xva_r - Xm_r) / Xs_r).to(device)
    Xsv_rn = ((Xsv_r - Xm_r) / Xs_r).to(device)
    Xnv_rn = ((Xnv_r - Xm_r) / Xs_r).to(device)
    ytr_rn = ((ytr_r - ym_r) / ys_r).to(device)
    yva_rn = ((yva_r - ym_r) / ys_r).to(device)

    residual_model = BaseNN(len(RESIDUAL_FEATURES)).to(device)
    residual_model, _ = train_generic(residual_model, Xtr_rn, ytr_rn, Xva_rn, yva_rn)

    def residual_trip_energy(rows_df, X_norm, model):
        model.eval()
        with torch.no_grad():
            residual_pred_W = (model(X_norm) * ys_r.to(device) + ym_r.to(device)).cpu().numpy()
        d = rows_df.copy()
        d["P_bat_pred_W"] = d["P_physics_W"] + residual_pred_W
        trip = (
            d.groupby(["VehId", "Trip"])
             .apply(
                 lambda g: pd.Series({
                     "E_true_Wh": np.sum(g["P_bat_out_W"] * g["dt_s"]) / 3600,
                     "E_pred_Wh": np.sum(g["P_bat_pred_W"] * g["dt_s"]) / 3600,
                     "n_samples": len(g),
                 }),
                 include_groups=False,
             )
             .reset_index()
        )
        return trip[trip["n_samples"] >= MIN_TRIP_SAMPLES].copy()

    residual_metrics = {
        "Train": evaluate_trip_table(residual_trip_energy(rows_tr_r, Xtr_rn, residual_model)),
        "Val": evaluate_trip_table(residual_trip_energy(rows_va_r, Xva_rn, residual_model)),
        "Test-SV": evaluate_trip_table(residual_trip_energy(rows_sv_r, Xsv_rn, residual_model)),
        "Test-NV": evaluate_trip_table(residual_trip_energy(rows_nv_r, Xnv_rn, residual_model)),
    }

    # ---- Final comparison table ----
    rows = []
    for split in ["Train", "Val", "Test-SV", "Test-NV"]:
        for model_name, metrics in [
            ("Physics-only", physics_metrics[split]),
            ("Data-NN", data_metrics[split]),
            (f"PINN (lambda={best_lambda})", pinn_results[best_lambda][split]),
            ("Residual NN", residual_metrics[split]),
        ]:
            rows.append({"Split": split, "Model": model_name, **metrics})

    summary_df = pd.DataFrame(rows).round(
        {"MAE_Wh": 1, "RMSE_Wh": 1, "R2": 4, "WAPE_pct": 2}
    )

    print("\n" + "=" * 70)
    print("FINAL COMPARATIVE RESULTS TABLE")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    summary_path = os.path.join(args.output_dir, "summary_results.csv")
    lambda_path = os.path.join(args.output_dir, "lambda_sweep.csv")
    summary_df.to_csv(summary_path, index=False)
    lambda_summary.to_csv(lambda_path, index=False)
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {lambda_path}")

    if not args.no_plots and HAS_MPL:
        plt.figure(figsize=(8, 5))
        plt.plot(lambda_summary["lambda"].astype(str), lambda_summary["Test-SV_WAPE"],
                  marker="o", label="Test-SV (held-out trips)")
        plt.plot(lambda_summary["lambda"].astype(str), lambda_summary["Test-NV_WAPE"],
                  marker="o", label="Test-NV (held-out vehicle)")
        plt.xlabel("Physics weight lambda")
        plt.ylabel("WAPE (%)")
        plt.title("Prediction Error vs. Physics Weight")
        plt.legend()
        plt.grid(axis="y", alpha=0.2)
        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "lambda_sensitivity.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Saved: {plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

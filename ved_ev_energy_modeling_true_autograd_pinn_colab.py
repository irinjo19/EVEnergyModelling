# -*- coding: utf-8 -*-
"""VED_EV_Energy_Modeling_TRUE_Autograd_PINN_Colab.ipynb


# VED EV Energy Modeling — Colab Version with Autograd PINN

This notebook compares **six approaches**:

1. Physics-only baseline  
2. Data-only neural network  
3. Fixed-physics consistency NN  
4. Learnable-physics NN (jointly learns `eta`, `Cd`, `Crr`, and effective mass)  
5. Residual neural network  
6. **Autograd PINN**: a differentiable velocity field is learned from the observed speed trace, and PyTorch `autograd` computes \(d\hat v/dt\). That derivative enters the governing vehicle-dynamics equation used in the physics loss.

The first five models are preserved from the earlier notebook. The sixth model is added specifically to test the derivative-based physics-informed formulation we discussed.

### Important implementation detail

VED contains many independent trips. A single function \(v(t)\) cannot represent all of them because every trip has its own velocity history. The Autograd PINN therefore uses a **trip-conditioned differentiable velocity network**:

\[
(t,\text{trip embedding}) \rightarrow \hat v(t)
\]

and obtains

\[
\frac{d\hat v}{dt}
\]

with `torch.autograd.grad`. The velocity network is trained only against the observed speed trace. For validation/test trips, its trip embeddings are fitted using speed only, **never battery-power labels**, so battery-energy evaluation remains held out.

This avoids pretending that a measured speed column is differentiable with respect to an independent time column.
"""

# Colab setup
# !pip -q install py7zr openpyxl

import os
print("Colab setup complete.")

import os
import random
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn

try:
    from IPython.display import display
except (ImportError, NameError):
    def display(*args, **kwargs):
        for arg in args:
            print(arg)

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
ETA_FIXED = 0.70        # drivetrain efficiency used by the fixed physics baseline/PINN

PINN_LAMBDAS = [0.1, 1.0, 5.0, 20.0]
PAPER_PINN_LAMBDAS = [0.1, 1.0, 5.0, 20.0]

# Bounds for physical parameters learned by the paper-style PINN.
# Bounding avoids unphysical parameter drift/identifiability failures.
ETA_BOUNDS = (0.50, 0.95)
CD_BOUNDS = (0.15, 0.50)
CRR_BOUNDS = (0.003, 0.030)
MASS_BOUNDS_KG = (0.80 * MASS_KG, 1.25 * MASS_KG)
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

"""## Step 1 - Download VED and retain the three pure-EV vehicles"""

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

"""## Step 2 - Leakage-safe Train / Val / Test-SV / Test-NV splits"""

# ============================================================
# Step 2 - Leakage-safe Train / Val / Test-SV / Test-NV splits
# ============================================================

def make_splits(ev_df, seed=SEED):
    test_nv_df = ev_df[ev_df["VehId"] == 541].copy()
    trainval_df = ev_df[ev_df["VehId"].isin([10, 455])].copy()
    trainval_df["veh_trip_key"] = list(zip(trainval_df["VehId"], trainval_df["Trip"]))

    rng = np.random.RandomState(seed)
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

"""## Step 3 - Preprocessing (identical for all four splits)"""

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

"""## Step 4 - Shared evaluation utilities"""

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
             })
         )
         .reset_index()
    )
    return trip[trip["n_samples"] >= MIN_TRIP_SAMPLES].copy()


def evaluate_trip_table(trip_df):
    return compute_metrics(trip_df["E_true_Wh"], trip_df["E_pred_Wh"])


def valid_rows(df):
    required = ["speed_ms_smooth", "accel_ms2", "P_aux_W", "OAT[DegC]", "P_bat_out_W", "dt_s"]
    return df[(df["speed_ms_smooth"] > MIN_SPEED_MS)].dropna(subset=required).copy()

"""## Step 5 - Physics-only baseline"""

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

"""## Step 6 - Models"""

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


def _inverse_bounded_sigmoid(value, low, high):
    """Map a physical initialization value into an unconstrained raw parameter."""
    p = (float(value) - low) / (high - low)
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return float(np.log(p / (1.0 - p)))


class PaperStylePINN(BaseNN):
    """Physics-informed NN with jointly learned physical parameters.

    This is closer to the uploaded EV-PINN paper than PINNFixedEta because the
    physics model is not a fixed target: eta, Cd, Crr, and effective mass are
    trainable parameters optimized together with the NN weights.

    Acceleration is still obtained from the measured speed trace during
    preprocessing (dv/dt by timestamp-aware differentiation). This keeps the
    comparison leakage-safe and practical for VED. 
    """

    def __init__(self):
        super().__init__(4)
        self.raw_eta = nn.Parameter(torch.tensor(
            _inverse_bounded_sigmoid(ETA_FIXED, *ETA_BOUNDS), dtype=torch.float32))
        self.raw_cd = nn.Parameter(torch.tensor(
            _inverse_bounded_sigmoid(CD, *CD_BOUNDS), dtype=torch.float32))
        self.raw_crr = nn.Parameter(torch.tensor(
            _inverse_bounded_sigmoid(CRR, *CRR_BOUNDS), dtype=torch.float32))
        self.raw_mass = nn.Parameter(torch.tensor(
            _inverse_bounded_sigmoid(MASS_KG, *MASS_BOUNDS_KG), dtype=torch.float32))

    @staticmethod
    def _bounded(raw, bounds):
        low, high = bounds
        return low + (high - low) * torch.sigmoid(raw)

    @property
    def eta(self):
        return self._bounded(self.raw_eta, ETA_BOUNDS)

    @property
    def cd(self):
        return self._bounded(self.raw_cd, CD_BOUNDS)

    @property
    def crr(self):
        return self._bounded(self.raw_crr, CRR_BOUNDS)

    @property
    def mass_kg(self):
        return self._bounded(self.raw_mass, MASS_BOUNDS_KG)

    def physics_pred_normalized(self, X_raw, y_mean, y_std):
        v, a, aux = X_raw[:, 0], X_raw[:, 1], X_raw[:, 2]
        F_roll = self.crr * self.mass_kg * G
        F_aero = 0.5 * RHO_AIR * self.cd * FRONTAL_AREA_M2 * v ** 2
        F_inertia = self.mass_kg * a
        P_wheel = (F_roll + F_aero + F_inertia) * v
        P_phys = P_wheel / self.eta + aux
        return (P_phys - y_mean.to(X_raw.device)) / y_std.to(X_raw.device)

    def physical_parameters(self):
        return {
            "eta": float(self.eta.detach().cpu()),
            "Cd": float(self.cd.detach().cpu()),
            "Crr": float(self.crr.detach().cpu()),
            "mass_kg": float(self.mass_kg.detach().cpu()),
            "rho_air_fixed": float(RHO_AIR),
            "frontal_area_m2_fixed": float(FRONTAL_AREA_M2),
        }


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

"""## 

The first run downloads the VED dataset from GitHub.

"""

def run_pipeline(data_dir='./ved_data', output_dir='./results', no_plots=False):

    os.makedirs(output_dir, exist_ok=True)
    set_all_seeds(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- Data ----
    ev_df = download_ved(data_dir)
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
    print("Validation-selected fixed-physics PINN lambda:", best_lambda)

    lambda_summary = pd.DataFrame({
        "lambda": [0.0] + PINN_LAMBDAS,
        "Test-SV_WAPE": [data_metrics["Test-SV"]["WAPE_pct"]] +
                         [pinn_results[l]["Test-SV"]["WAPE_pct"] for l in PINN_LAMBDAS],
        "Test-NV_WAPE": [data_metrics["Test-NV"]["WAPE_pct"]] +
                         [pinn_results[l]["Test-NV"]["WAPE_pct"] for l in PINN_LAMBDAS],
    })

    # ---- Learnable-physics NN: jointly learn selected physical parameters ----
    print("\nTraining learnable-physics NN...")
    paper_pinn_results = {}
    paper_pinn_params = {}
    paper_pinn_models = {}

    for lam in PAPER_PINN_LAMBDAS:
        print(f"Training Learnable-physics NN: lambda={lam}")
        model = PaperStylePINN().to(device)
        physics_fn = lambda m, Xn: m.physics_pred_normalized(X_train_raw, y_mean, y_std)
        model, _ = train_generic(
            model, X_train_n, y_train_n, X_val_n, y_val_n,
            physics_term_fn=physics_fn, lambda_physics=lam,
        )
        paper_pinn_models[lam] = model
        paper_pinn_results[lam] = eval_model_all_splits(model, y_mean, y_std)
        paper_pinn_params[lam] = model.physical_parameters()
        print("  learned parameters:", paper_pinn_params[lam])

    best_paper_lambda = min(
        PAPER_PINN_LAMBDAS,
        key=lambda l: paper_pinn_results[l]["Val"]["WAPE_pct"]
    )
    print("Validation-selected paper-style PINN lambda:", best_paper_lambda)
    print("Selected learned physical parameters:", paper_pinn_params[best_paper_lambda])

    paper_lambda_summary = pd.DataFrame([
        {
            "lambda": lam,
            "Val_WAPE": paper_pinn_results[lam]["Val"]["WAPE_pct"],
            "Test-SV_WAPE": paper_pinn_results[lam]["Test-SV"]["WAPE_pct"],
            "Test-NV_WAPE": paper_pinn_results[lam]["Test-NV"]["WAPE_pct"],
            **paper_pinn_params[lam],
        }
        for lam in PAPER_PINN_LAMBDAS
    ])

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
                 })
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
            (f"Fixed-physics PINN (lambda={best_lambda})", pinn_results[best_lambda][split]),
            (f"Learnable-physics NN (lambda={best_paper_lambda})", paper_pinn_results[best_paper_lambda][split]),
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

    summary_path = os.path.join(output_dir, "summary_results.csv")
    lambda_path = os.path.join(output_dir, "lambda_sweep.csv")
    paper_lambda_path = os.path.join(output_dir, "paper_style_pinn_sweep.csv")
    summary_df.to_csv(summary_path, index=False)
    lambda_summary.to_csv(lambda_path, index=False)
    paper_lambda_summary.to_csv(paper_lambda_path, index=False)
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {lambda_path}")
    print(f"Saved: {paper_lambda_path}")

    if not no_plots and HAS_MPL:
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
        plot_path = os.path.join(output_dir, "lambda_sensitivity.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Saved: {plot_path}")

    print("\nDone.")

    return {
        'summary_df': summary_df,
        'lambda_summary': lambda_summary,
        'paper_lambda_summary': paper_lambda_summary,
        'best_fixed_lambda': best_lambda,
        'best_paper_lambda': best_paper_lambda,
        'best_paper_parameters': paper_pinn_params[best_paper_lambda],
        # Return preprocessed splits so the Autograd PINN add-on can reuse them
        # without downloading/extracting/preprocessing VED a second time.
        'train_df': train_df,
        'val_df': val_df,
        'test_sv_df': test_sv_df,
        'test_nv_df': test_nv_df,
    }

# Run all five models
results = run_pipeline(
    data_dir="./ved_data",
    output_dir="./results",
    no_plots=False,
)

"""## Fixed vs learned physical parameters

The table below compares the original fixed parameters against the values obtained by the learnable-physics NN for each \(\lambda\).

This is important because the validation-selected \(\lambda=0.1\) solution gave the best prediction among that model's lambda sweep, but its inferred physical parameters moved substantially away from the fixed values. In contrast, \(\lambda=5\) and \(20\) stay much closer to the original physical values.

"""

fixed_parameter_values = {
    "eta": ETA_FIXED,
    "Cd": CD,
    "Crr": CRR,
    "mass_kg": MASS_KG,
}

learned_parameter_rows = []
for _, row in results["paper_lambda_summary"].iterrows():
    lam = float(row["lambda"])
    learned_parameter_rows.append({
        "lambda": lam,
        "eta": row["eta"],
        "eta_change_pct": 100 * (row["eta"] / ETA_FIXED - 1),
        "Cd": row["Cd"],
        "Cd_change_pct": 100 * (row["Cd"] / CD - 1),
        "Crr": row["Crr"],
        "Crr_change_pct": 100 * (row["Crr"] / CRR - 1),
        "mass_kg": row["mass_kg"],
        "mass_change_pct": 100 * (row["mass_kg"] / MASS_KG - 1),
        "Val_WAPE": row["Val_WAPE"],
    })

parameter_comparison_df = pd.DataFrame(learned_parameter_rows)

print("Fixed parameters:")
for k, v in fixed_parameter_values.items():
    print(f"  {k}: {v}")

display(parameter_comparison_df.round({
    "eta": 4, "eta_change_pct": 1,
    "Cd": 4, "Cd_change_pct": 1,
    "Crr": 5, "Crr_change_pct": 1,
    "mass_kg": 1, "mass_change_pct": 1,
    "Val_WAPE": 2,
}))

"""# Derivative-based Autograd PINN


For the Autograd PINN, the differentiable speed field is

\[
\hat v = f_\phi(\tau, z_{\rm trip}),
\]

where \(\tau\) is normalized time within a trip and \(z_{\rm trip}\) is a learned trip embedding.

PyTorch computes

\[
\frac{d\hat v}{dt}
=
\frac{d\hat v}{d\tau}\frac{d\tau}{dt}
\]

using `torch.autograd.grad`.

The governing power model is

\[
P_{\rm phys}
=
\frac{1}{\eta}
\left[
\frac12\rho A C_d\hat v^3
+
C_{rr}mg\hat v
+
m\hat v\frac{d\hat v}{dt}
\right]
+
P_{\rm aux}.
\]

The total training loss is

\[
L =
L_{\rm power}
+
\alpha_v L_{\rm speed}
+
\lambda L_{\rm physics}.
\]

For this first clean experiment, the Autograd PINN keeps the physical parameters fixed. That isolates the effect of using a differentiable state and an actual derivative in the physics branch. 

"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Autograd PINN device:", device)

# ============================================================
# AUTOGRAD PINN
# ============================================================

import copy
import math

AUTOGRAD_PINN_LAMBDAS = [0.1, 1.0, 5.0, 20.0]

# Weight on the velocity-reconstruction loss.
SPEED_LOSS_WEIGHT = 1.0

# These are deliberately moderate so the notebook is practical in Colab.
# Increase them later for a final publication run if desired.
SPEED_PRETRAIN_STEPS = 350
AUTOGRAD_PINN_STEPS = 500
AUTOGRAD_BATCH_SIZE = 8192
EVAL_CHUNK_SIZE = 32768


def prepare_autograd_rows(df):
    """
    Build tensors for the derivative-based PINN.

    Each trip gets:
      * tau in [0, 1]
      * its own trip index
      * trip duration in seconds

    The battery-power label is NOT used to construct tau or the trip embedding.
    """
    rows = valid_rows(df).copy()
    rows.sort_values(["VehId", "Trip", "Timestamp(ms)"], inplace=True)
    rows.reset_index(drop=True, inplace=True)

    key = list(zip(rows["VehId"].astype(int), rows["Trip"]))
    unique_keys = {k: i for i, k in enumerate(dict.fromkeys(key))}
    rows["autograd_trip_id"] = [unique_keys[k] for k in key]

    t0 = rows.groupby("autograd_trip_id")["Timestamp(ms)"].transform("min")
    t1 = rows.groupby("autograd_trip_id")["Timestamp(ms)"].transform("max")
    rows["t_rel_s"] = (rows["Timestamp(ms)"] - t0) / 1000.0
    rows["trip_duration_s"] = ((t1 - t0) / 1000.0).clip(lower=1e-3)
    rows["tau"] = rows["t_rel_s"] / rows["trip_duration_s"]

    data = {
        "rows": rows,
        "tau": torch.tensor(rows["tau"].to_numpy(), dtype=torch.float32).view(-1, 1),
        "trip_id": torch.tensor(rows["autograd_trip_id"].to_numpy(), dtype=torch.long),
        "duration": torch.tensor(rows["trip_duration_s"].to_numpy(), dtype=torch.float32).view(-1, 1),
        "v_true": torch.tensor(rows["speed_ms_smooth"].to_numpy(), dtype=torch.float32).view(-1, 1),
        "aux": torch.tensor(rows["P_aux_W"].to_numpy(), dtype=torch.float32).view(-1, 1),
        "oat": torch.tensor(rows["OAT[DegC]"].to_numpy(), dtype=torch.float32).view(-1, 1),
        "p_true": torch.tensor(rows["P_bat_out_W"].to_numpy(), dtype=torch.float32).view(-1, 1),
        "n_trips": len(unique_keys),
    }
    return data


class TripConditionedVelocityNet(nn.Module):
    """
    Differentiable velocity field:
        (normalized trip time, learned trip embedding) -> normalized velocity

    Because tau is an actual differentiable input, autograd can compute
    d(v_hat)/d(tau), which is converted to physical d(v_hat)/dt.
    """
    def __init__(self, n_trips, embedding_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(n_trips, embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(1 + embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, tau, trip_id):
        z = self.embedding(trip_id)
        return self.net(torch.cat([tau, z], dim=1))


class AutogradPowerNet(nn.Module):
    """Battery-power network used by the derivative-based PINN."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


# Reuse the already preprocessed splits from the original pipeline.
ag_train = prepare_autograd_rows(results["train_df"])
ag_val = prepare_autograd_rows(results["val_df"])
ag_sv = prepare_autograd_rows(results["test_sv_df"])
ag_nv = prepare_autograd_rows(results["test_nv_df"])

# Normalization is fitted on training data only.
ag_v_mean = ag_train["v_true"].mean().to(device)
ag_v_std = ag_train["v_true"].std().clamp_min(1e-8).to(device)

ag_y_mean = ag_train["p_true"].mean().to(device)
ag_y_std = ag_train["p_true"].std().clamp_min(1e-8).to(device)

# Power-net input normalization: [v_hat, a_autograd, P_aux, OAT].
# For acceleration scaling ONLY, use the training-set measured acceleration
# statistics. The Autograd PINN itself will not use measured acceleration.
ag_x_mean = torch.tensor([
    results["train_df"].pipe(valid_rows)["speed_ms_smooth"].mean(),
    results["train_df"].pipe(valid_rows)["accel_ms2"].mean(),
    results["train_df"].pipe(valid_rows)["P_aux_W"].mean(),
    results["train_df"].pipe(valid_rows)["OAT[DegC]"].mean(),
], dtype=torch.float32, device=device)

ag_x_std = torch.tensor([
    results["train_df"].pipe(valid_rows)["speed_ms_smooth"].std(),
    results["train_df"].pipe(valid_rows)["accel_ms2"].std(),
    results["train_df"].pipe(valid_rows)["P_aux_W"].std(),
    results["train_df"].pipe(valid_rows)["OAT[DegC]"].std(),
], dtype=torch.float32, device=device).clamp_min(1e-8)


def random_batch_indices(n, batch_size, device):
    size = min(batch_size, n)
    return torch.randint(0, n, (size,), device=device)


def pretrain_velocity_field(data, steps=SPEED_PRETRAIN_STEPS, lr=3e-3):
    """
    Fit a differentiable v(t) field using SPEED ONLY.

    This is also how validation/test trip embeddings are fitted, so no
    battery-power target is used in the trajectory representation.
    """
    model = TripConditionedVelocityNet(data["n_trips"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = len(data["tau"])
    tau_all = data["tau"].to(device)
    trip_all = data["trip_id"].to(device)
    v_all_n = ((data["v_true"].to(device) - ag_v_mean) / ag_v_std)

    model.train()
    for step in range(steps):
        idx = random_batch_indices(n, AUTOGRAD_BATCH_SIZE, device)
        pred_v_n = model(tau_all[idx], trip_all[idx])
        loss = torch.mean((pred_v_n - v_all_n[idx]) ** 2)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 100 == 0:
            print(f"    speed pretrain step {step+1:4d}/{steps}: MSE={loss.item():.5f}")

    return model


def velocity_and_autograd_accel(speed_model, tau, trip_id, duration, create_graph):
    """
    Compute v_hat and dv_hat/dt using torch.autograd.grad.

    tau = t / duration, therefore:
        dv/dt = (dv/dtau) / duration
    after undoing velocity normalization.
    """
    tau_leaf = tau.detach().clone().requires_grad_(True)

    v_hat_n = speed_model(tau_leaf, trip_id)

    dvn_dtau = torch.autograd.grad(
        outputs=v_hat_n,
        inputs=tau_leaf,
        grad_outputs=torch.ones_like(v_hat_n),
        create_graph=create_graph,
        retain_graph=create_graph,
        only_inputs=True,
    )[0]

    v_hat = v_hat_n * ag_v_std + ag_v_mean
    a_hat = (dvn_dtau * ag_v_std) / duration.clamp_min(1e-3)

    return v_hat_n, v_hat, a_hat


def fixed_physics_from_autograd(v_hat, a_hat, aux):
    """
    Governing vehicle power equation using AUTOGRAD acceleration.
    No precomputed accel_ms2 enters this calculation.
    """
    F_roll = CRR * MASS_KG * G
    F_aero = 0.5 * RHO_AIR * CD * FRONTAL_AREA_M2 * v_hat ** 2
    F_inertia = MASS_KG * a_hat
    P_wheel = (F_roll + F_aero + F_inertia) * v_hat
    return P_wheel / ETA_FIXED + aux


def train_autograd_pinn(train_data, base_speed_state, lam,
                        steps=AUTOGRAD_PINN_STEPS, lr=3e-3):
    """
    Jointly optimize:
      * train-trip differentiable velocity field
      * battery-power NN

    Loss:
      L = L_power + SPEED_LOSS_WEIGHT*L_speed + lam*L_physics
    """
    speed_model = TripConditionedVelocityNet(train_data["n_trips"]).to(device)
    speed_model.load_state_dict(copy.deepcopy(base_speed_state))
    power_model = AutogradPowerNet().to(device)

    opt = torch.optim.Adam(
        list(speed_model.parameters()) + list(power_model.parameters()),
        lr=lr,
    )

    n = len(train_data["tau"])

    tau_all = train_data["tau"].to(device)
    trip_all = train_data["trip_id"].to(device)
    dur_all = train_data["duration"].to(device)
    v_true_all = train_data["v_true"].to(device)
    aux_all = train_data["aux"].to(device)
    oat_all = train_data["oat"].to(device)
    p_true_all = train_data["p_true"].to(device)

    best_state = None
    best_loss = float("inf")

    speed_model.train()
    power_model.train()

    for step in range(steps):
        idx = random_batch_indices(n, AUTOGRAD_BATCH_SIZE, device)

        tau = tau_all[idx]
        trip_id = trip_all[idx]
        duration = dur_all[idx]
        v_true = v_true_all[idx]
        aux = aux_all[idx]
        oat = oat_all[idx]
        p_true = p_true_all[idx]

        v_hat_n, v_hat, a_hat = velocity_and_autograd_accel(
            speed_model, tau, trip_id, duration, create_graph=True
        )

        x_phys = torch.cat([v_hat, a_hat, aux, oat], dim=1)
        x_n = (x_phys - ag_x_mean) / ag_x_std

        p_hat_n = power_model(x_n)
        p_true_n = (p_true - ag_y_mean) / ag_y_std

        p_phys = fixed_physics_from_autograd(v_hat, a_hat, aux)
        p_phys_n = (p_phys - ag_y_mean) / ag_y_std

        v_true_n = (v_true - ag_v_mean) / ag_v_std

        power_loss = torch.mean((p_hat_n - p_true_n) ** 2)
        speed_loss = torch.mean((v_hat_n - v_true_n) ** 2)
        physics_loss = torch.mean((p_hat_n - p_phys_n) ** 2)

        loss = power_loss + SPEED_LOSS_WEIGHT * speed_loss + lam * physics_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(speed_model.parameters()) + list(power_model.parameters()),
            max_norm=10.0
        )
        opt.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {
                "speed": copy.deepcopy(speed_model.state_dict()),
                "power": copy.deepcopy(power_model.state_dict()),
            }

        if (step + 1) % 100 == 0:
            print(
                f"    step {step+1:4d}/{steps}: "
                f"total={loss.item():.5f}, "
                f"power={power_loss.item():.5f}, "
                f"speed={speed_loss.item():.5f}, "
                f"physics={physics_loss.item():.5f}"
            )

    speed_model.load_state_dict(best_state["speed"])
    power_model.load_state_dict(best_state["power"])
    return speed_model, power_model


def predict_autograd_pinn(power_model, speed_model, data, chunk_size=EVAL_CHUNK_SIZE):
    """
    Predict battery power while computing acceleration via autograd.
    """
    speed_model.eval()
    power_model.eval()

    preds = []

    n = len(data["tau"])
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        tau = data["tau"][start:end].to(device)
        trip_id = data["trip_id"][start:end].to(device)
        duration = data["duration"][start:end].to(device)
        aux = data["aux"][start:end].to(device)
        oat = data["oat"][start:end].to(device)

        # We cannot use torch.no_grad here because dv/dt itself requires autograd.
        v_hat_n, v_hat, a_hat = velocity_and_autograd_accel(
            speed_model, tau, trip_id, duration, create_graph=False
        )

        x = torch.cat([v_hat, a_hat, aux, oat], dim=1)
        x_n = (x - ag_x_mean) / ag_x_std

        with torch.no_grad():
            p_hat = power_model(x_n) * ag_y_std + ag_y_mean

        preds.append(p_hat.detach().cpu().numpy().reshape(-1))

    return np.concatenate(preds)


def evaluate_autograd_model(power_model, speed_models, datasets):
    out = {}
    for split_name, data in datasets.items():
        p_pred = predict_autograd_pinn(
            power_model,
            speed_models[split_name],
            data,
        )
        trip = integrate_trip_energy(data["rows"], p_pred)
        out[split_name] = evaluate_trip_table(trip)
    return out


print("\nPretraining differentiable velocity fields from SPEED ONLY...")
print("  Train:")
base_speed_train = pretrain_velocity_field(ag_train)
print("  Validation:")
base_speed_val = pretrain_velocity_field(ag_val)
print("  Test-SV:")
base_speed_sv = pretrain_velocity_field(ag_sv)
print("  Test-NV:")
base_speed_nv = pretrain_velocity_field(ag_nv)

base_train_state = copy.deepcopy(base_speed_train.state_dict())

autograd_pinn_results = {}
autograd_pinn_models = {}

for lam in AUTOGRAD_PINN_LAMBDAS:
    print(f"\nTraining AUTOGRAD PINN: lambda={lam}")

    trained_speed, trained_power = train_autograd_pinn(
        ag_train,
        base_train_state,
        lam,
    )

    split_speed_models = {
        "Train": trained_speed,
        "Val": base_speed_val,
        "Test-SV": base_speed_sv,
        "Test-NV": base_speed_nv,
    }
    split_data = {
        "Train": ag_train,
        "Val": ag_val,
        "Test-SV": ag_sv,
        "Test-NV": ag_nv,
    }

    metrics = evaluate_autograd_model(
        trained_power,
        split_speed_models,
        split_data,
    )

    autograd_pinn_results[lam] = metrics
    autograd_pinn_models[lam] = (trained_speed, trained_power)

    print(
        f"  Val WAPE={metrics['Val']['WAPE_pct']:.2f}% | "
        f"Test-SV={metrics['Test-SV']['WAPE_pct']:.2f}% | "
        f"Test-NV={metrics['Test-NV']['WAPE_pct']:.2f}%"
    )

best_autograd_lambda = min(
    AUTOGRAD_PINN_LAMBDAS,
    key=lambda l: autograd_pinn_results[l]["Val"]["WAPE_pct"],
)

print("\nValidation-selected AUTOGRAD PINN lambda:", best_autograd_lambda)

autograd_lambda_summary = pd.DataFrame([
    {
        "lambda": lam,
        "Train_WAPE": autograd_pinn_results[lam]["Train"]["WAPE_pct"],
        "Val_WAPE": autograd_pinn_results[lam]["Val"]["WAPE_pct"],
        "Test_SV_WAPE": autograd_pinn_results[lam]["Test-SV"]["WAPE_pct"],
        "Test_NV_WAPE": autograd_pinn_results[lam]["Test-NV"]["WAPE_pct"],
    }
    for lam in AUTOGRAD_PINN_LAMBDAS
])

display(autograd_lambda_summary)

# ============================================================
# FINAL SIX-MODEL COMPARISON
# ============================================================

best_ag = autograd_pinn_results[best_autograd_lambda]

autograd_rows = []
for split in ["Train", "Val", "Test-SV", "Test-NV"]:
    autograd_rows.append({
        "Split": split,
        "Model": f"Autograd PINN (lambda={best_autograd_lambda})",
        **best_ag[split],
    })

autograd_df = pd.DataFrame(autograd_rows)

combined_summary_df = pd.concat(
    [results["summary_df"], autograd_df],
    ignore_index=True
)

combined_summary_df = combined_summary_df.round(
    {"MAE_Wh": 1, "RMSE_Wh": 1, "R2": 4, "WAPE_pct": 2}
)

print("=" * 90)
print("FINAL COMPARATIVE RESULTS — INCLUDING DERIVATIVE-BASED AUTOGRAD PINN")
print("=" * 90)
display(combined_summary_df)

combined_summary_path = "./results/summary_results_with_autograd_pinn.csv"
autograd_lambda_path = "./results/autograd_pinn_lambda_sweep.csv"

combined_summary_df.to_csv(combined_summary_path, index=False)
autograd_lambda_summary.to_csv(autograd_lambda_path, index=False)

print("Saved:", combined_summary_path)
print("Saved:", autograd_lambda_path)

"""## Original five-model comparison (before Autograd PINN)"""

from IPython.display import display

display(results["summary_df"])

"""## Learned physical parameters from the learnable-physics NN"""

print("Best lambda:", results["best_paper_lambda"])
print("\nLearned physical parameters:")
for k, v in results["best_paper_parameters"].items():
    print(f"{k}: {v}")

display(results["paper_lambda_summary"])

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt

# ============================================================
# VISUALIZE FINAL RESULTS
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Plot 1: WAPE comparison across all models and splits
# ------------------------------------------------------------

plot_df = combined_summary_df.copy()

pivot_wape = plot_df.pivot(
    index="Model",
    columns="Split",
    values="WAPE_pct"
)

# Keep split ordering consistent
split_order = ["Train", "Val", "Test-SV", "Test-NV"]
pivot_wape = pivot_wape[
    [c for c in split_order if c in pivot_wape.columns]
]

ax = pivot_wape.plot(
    kind="bar",
    figsize=(14, 6)
)

ax.set_ylabel("WAPE (%)")
ax.set_xlabel("")
ax.set_title("Energy Prediction Error by Model")
ax.legend(title="Dataset Split")

plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Plot 2: MAE comparison
# ------------------------------------------------------------

pivot_mae = plot_df.pivot(
    index="Model",
    columns="Split",
    values="MAE_Wh"
)

pivot_mae = pivot_mae[
    [c for c in split_order if c in pivot_mae.columns]
]

ax = pivot_mae.plot(
    kind="bar",
    figsize=(14, 6)
)

ax.set_ylabel("Trip Energy MAE (Wh)")
ax.set_xlabel("")
ax.set_title("Trip Energy MAE by Model")
ax.legend(title="Dataset Split")

plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Plot 3: R² comparison
# ------------------------------------------------------------

pivot_r2 = plot_df.pivot(
    index="Model",
    columns="Split",
    values="R2"
)

pivot_r2 = pivot_r2[
    [c for c in split_order if c in pivot_r2.columns]
]

ax = pivot_r2.plot(
    kind="bar",
    figsize=(14, 6)
)

ax.set_ylabel("R²")
ax.set_xlabel("")
ax.set_title("Trip Energy Prediction R² by Model")
ax.legend(title="Dataset Split")

plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Plot 4: Autograd PINN lambda sensitivity
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    autograd_lambda_summary["lambda"],
    autograd_lambda_summary["Train_WAPE"],
    marker="o",
    label="Train"
)

plt.plot(
    autograd_lambda_summary["lambda"],
    autograd_lambda_summary["Val_WAPE"],
    marker="o",
    label="Validation"
)

plt.plot(
    autograd_lambda_summary["lambda"],
    autograd_lambda_summary["Test_SV_WAPE"],
    marker="o",
    label="Test-SV"
)

plt.plot(
    autograd_lambda_summary["lambda"],
    autograd_lambda_summary["Test_NV_WAPE"],
    marker="o",
    label="Test-NV"
)

plt.xscale("log")

plt.xlabel("Physics Loss Weight λ")
plt.ylabel("WAPE (%)")
plt.title("Autograd PINN Sensitivity to Physics Weight λ")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Plot 5: Fixed vs learned physical parameters
# ------------------------------------------------------------

display(parameter_comparison_df)

print("\nBest Autograd PINN lambda:", best_autograd_lambda)
print("\nFinal model comparison:")
display(
    combined_summary_df.sort_values(
        ["Split", "WAPE_pct"]
    )
)

import matplotlib
import matplotlib.pyplot as plt

# ============================================================
# FINAL RESULT VISUALIZATIONS
# ============================================================

import os
import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("./results", exist_ok=True)

plot_df = combined_summary_df.copy()

split_order = ["Train", "Val", "Test-SV", "Test-NV"]


# ============================================================
# 1. WAPE
# ============================================================

pivot_wape = plot_df.pivot(
    index="Model",
    columns="Split",
    values="WAPE_pct"
)

pivot_wape = pivot_wape[
    [s for s in split_order if s in pivot_wape.columns]
]

fig, ax = plt.subplots(figsize=(14, 7))

pivot_wape.plot(
    kind="bar",
    ax=ax
)

ax.set_title("WAPE Comparison Across Models")
ax.set_ylabel("WAPE (%)")
ax.set_xlabel("")
ax.legend(title="Split")

plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig(
    "./results/model_wape_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 2. MAE
# ============================================================

pivot_mae = plot_df.pivot(
    index="Model",
    columns="Split",
    values="MAE_Wh"
)

pivot_mae = pivot_mae[
    [s for s in split_order if s in pivot_mae.columns]
]

fig, ax = plt.subplots(figsize=(14, 7))

pivot_mae.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Trip Energy MAE Across Models")
ax.set_ylabel("MAE (Wh)")
ax.set_xlabel("")
ax.legend(title="Split")

plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig(
    "./results/model_mae_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 3. R²
# ============================================================

pivot_r2 = plot_df.pivot(
    index="Model",
    columns="Split",
    values="R2"
)

pivot_r2 = pivot_r2[
    [s for s in split_order if s in pivot_r2.columns]
]

fig, ax = plt.subplots(figsize=(14, 7))

pivot_r2.plot(
    kind="bar",
    ax=ax
)

ax.set_title("R² Comparison Across Models")
ax.set_ylabel("R²")
ax.set_xlabel("")
ax.legend(title="Split")

plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig(
    "./results/model_r2_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 4. AUTOGRAD PINN LAMBDA SENSITIVITY
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(
    autograd_lambda_summary["lambda"],
    autograd_lambda_summary["Train_WAPE"],
    marker="o",
    label="Train"
)

ax.plot(
    autograd_lambda_summary["lambda"],
    autograd_lambda_summary["Val_WAPE"],
    marker="o",
    label="Validation"
)

ax.plot(
    autograd_lambda_summary["lambda"],
    autograd_lambda_summary["Test_SV_WAPE"],
    marker="o",
    label="Test-SV"
)

ax.plot(
    autograd_lambda_summary["lambda"],
    autograd_lambda_summary["Test_NV_WAPE"],
    marker="o",
    label="Test-NV"
)

ax.set_xscale("log")

ax.set_xlabel("Physics Loss Weight λ")
ax.set_ylabel("WAPE (%)")
ax.set_title("Autograd PINN Sensitivity to Physics Weight")

ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "./results/autograd_pinn_lambda_sensitivity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("Plots saved in ./results/")

"""## Download the result files"""

# Zip all CSVs and plots, including the Autograd PINN outputs
import shutil
try:
    from google.colab import files as colab_files
except ImportError:
    colab_files = None

zip_path = shutil.make_archive("./ved_energy_results_with_autograd_pinn", "zip", "./results")
print("Created:", zip_path)

# Uncomment when you want to download:
# colab_files.download(zip_path)

# ============================================================
# DATA EFFICIENCY EXPERIMENT
# Data-NN vs Fixed-Physics PINN
# ============================================================

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Fractions of the ORIGINAL TRAINING SET
TRAIN_FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]

# Use the lambda already selected by the main experiment.
DATA_SWEEP_PINN_LAMBDA = results["best_fixed_lambda"]

print("Fixed-PINN lambda:", DATA_SWEEP_PINN_LAMBDA)

# ------------------------------------------------------------
# Fixed evaluation sets
# ------------------------------------------------------------

full_train_df = results["train_df"]
val_df       = results["val_df"]
test_sv_df   = results["test_sv_df"]
test_nv_df   = results["test_nv_df"]


# ------------------------------------------------------------
# Sample WHOLE TRIPS rather than individual rows
# ------------------------------------------------------------

def sample_training_trips(df, fraction, seed=SEED):

    trip_table = (
        df[["VehId", "Trip"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Randomize trip order ONCE deterministically.
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(trip_table))
    trip_table = trip_table.iloc[order].reset_index(drop=True)

    n_keep = max(
        1,
        int(np.ceil(len(trip_table) * fraction))
    )

    selected = trip_table.iloc[:n_keep]

    sampled = df.merge(
        selected,
        on=["VehId", "Trip"],
        how="inner"
    )

    return sampled


# ------------------------------------------------------------
# Tensor helper
#
# This was local to run_pipeline() in the original notebook,
# so we define an equivalent helper here.
# ------------------------------------------------------------

def sweep_make_tensors(df):

    rows = valid_rows(df).copy()

    X = torch.tensor(
        rows[FEATURES].to_numpy(),
        dtype=torch.float32
    )

    y = torch.tensor(
        rows["P_bat_out_W"].to_numpy(),
        dtype=torch.float32
    )

    return X, y, rows


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

data_size_results = []


for frac in TRAIN_FRACTIONS:

    print("\n" + "=" * 72)
    print(f"TRAINING FRACTION = {frac:.0%}")
    print("=" * 72)

    # --------------------------------------------------------
    # Reduced training set
    # --------------------------------------------------------

    train_subset = sample_training_trips(
        full_train_df,
        fraction=frac,
        seed=SEED
    )

    n_train_trips = (
        train_subset[["VehId", "Trip"]]
        .drop_duplicates()
        .shape[0]
    )

    print(f"Training rows:  {len(train_subset):,}")
    print(f"Training trips: {n_train_trips}")


    # --------------------------------------------------------
    # Construct tensors
    # --------------------------------------------------------

    X_train_t, y_train_t, train_rows = sweep_make_tensors(
        train_subset
    )

    X_val_t, y_val_t, val_rows = sweep_make_tensors(
        val_df
    )

    X_sv_t, y_sv_t, sv_rows = sweep_make_tensors(
        test_sv_df
    )

    X_nv_t, y_nv_t, nv_rows = sweep_make_tensors(
        test_nv_df
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Re-fit normalization using ONLY the reduced training set.
    # --------------------------------------------------------

    X_mean = X_train_t.mean(0)
    X_std = X_train_t.std(0).clamp_min(1e-8)

    y_mean = y_train_t.mean()
    y_std = y_train_t.std().clamp_min(1e-8)


    def sweep_norm_X(X):
        return ((X - X_mean) / X_std).to(device)


    X_train_n = sweep_norm_X(X_train_t)
    X_val_n   = sweep_norm_X(X_val_t)
    X_sv_n    = sweep_norm_X(X_sv_t)
    X_nv_n    = sweep_norm_X(X_nv_t)

    y_train_n = (
        (y_train_t - y_mean) / y_std
    ).to(device)

    y_val_n = (
        (y_val_t - y_mean) / y_std
    ).to(device)

    X_train_raw = X_train_t.to(device)


    # --------------------------------------------------------
    # Evaluation helper
    # --------------------------------------------------------

    def evaluate_sweep_model(model):

        return {

            "Val":
                evaluate_trip_table(
                    integrate_trip_energy(
                        val_rows,
                        predict_power(
                            model,
                            X_val_n,
                            y_mean,
                            y_std
                        )
                    )
                ),

            "Test-SV":
                evaluate_trip_table(
                    integrate_trip_energy(
                        sv_rows,
                        predict_power(
                            model,
                            X_sv_n,
                            y_mean,
                            y_std
                        )
                    )
                ),

            "Test-NV":
                evaluate_trip_table(
                    integrate_trip_energy(
                        nv_rows,
                        predict_power(
                            model,
                            X_nv_n,
                            y_mean,
                            y_std
                        )
                    )
                )
        }


    # ========================================================
    # MODEL 1: DATA-NN
    # ========================================================

    print("\nTraining Data-NN...")

    set_all_seeds(SEED)

    data_model = BaseNN(
        len(FEATURES)
    ).to(device)

    data_model, _ = train_generic(
        data_model,
        X_train_n,
        y_train_n,
        X_val_n,
        y_val_n
    )

    data_metrics = evaluate_sweep_model(
        data_model
    )


    # ========================================================
    # MODEL 2: FIXED-PHYSICS PINN
    # ========================================================

    print(
        f"Training Fixed-physics PINN "
        f"(lambda={DATA_SWEEP_PINN_LAMBDA})..."
    )

    set_all_seeds(SEED)

    pinn_model = PINNFixedEta().to(device)


    def physics_term_fn(model, Xn):

        return model.physics_pred_normalized(
            X_train_raw,
            y_mean,
            y_std
        )


    pinn_model, _ = train_generic(
        pinn_model,
        X_train_n,
        y_train_n,
        X_val_n,
        y_val_n,
        physics_term_fn=physics_term_fn,
        lambda_physics=DATA_SWEEP_PINN_LAMBDA
    )

    pinn_metrics = evaluate_sweep_model(
        pinn_model
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    for model_name, metrics in [

        ("Data-NN", data_metrics),

        (
            f"Fixed-physics PINN "
            f"(lambda={DATA_SWEEP_PINN_LAMBDA})",
            pinn_metrics
        )

    ]:

        for split in [
            "Val",
            "Test-SV",
            "Test-NV"
        ]:

            data_size_results.append({

                "train_fraction": frac,

                "train_percent":
                    frac * 100,

                "train_rows":
                    len(train_subset),

                "train_trips":
                    n_train_trips,

                "model":
                    model_name,

                "split":
                    split,

                **metrics[split]
            })


    # Quick output after every fraction
    print("\nWAPE results:")

    print(
        f"  Data-NN: "
        f"SV={data_metrics['Test-SV']['WAPE_pct']:.2f}% | "
        f"NV={data_metrics['Test-NV']['WAPE_pct']:.2f}%"
    )

    print(
        f"  Fixed PINN: "
        f"SV={pinn_metrics['Test-SV']['WAPE_pct']:.2f}% | "
        f"NV={pinn_metrics['Test-NV']['WAPE_pct']:.2f}%"
    )


# ============================================================
# FINAL TABLE
# ============================================================

data_size_df = pd.DataFrame(
    data_size_results
)

display(data_size_df)

# Save
data_size_df.to_csv(
    "./results/data_size_sweep.csv",
    index=False
)

print(
    "\nSaved: "
    "./results/data_size_sweep.csv"
)

# Commented out IPython magic to ensure Python compatibility.
# ============================================================
# PLOT DATA EFFICIENCY RESULTS
# ============================================================

# %matplotlib inline
import matplotlib.pyplot as plt


for split_name in [
    "Val",
    "Test-SV",
    "Test-NV"
]:

    subset = data_size_df[
        data_size_df["split"] == split_name
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for model_name in subset["model"].unique():

        m = (
            subset[
                subset["model"] == model_name
            ]
            .sort_values("train_percent")
        )

        ax.plot(
            m["train_percent"],
            m["WAPE_pct"],
            marker="o",
            linewidth=2,
            label=model_name
        )

    ax.set_xlabel(
        "Training Data Used (%)"
    )

    ax.set_ylabel(
        "WAPE (%)"
    )

    ax.set_title(
        f"Data Efficiency — {split_name}"
    )

    ax.set_xticks(
        [5, 10, 25, 50, 100]
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    plt.tight_layout()

    path = (
        f"./results/"
        f"data_efficiency_{split_name}.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("Saved:", path)
# EV Energy Consumption Modeling: Physics-Based, Data-Driven, Residual-Learning, and PINN Approaches

Code for a research project comparing four approaches to predicting electric
vehicle battery energy consumption from real-world telemetry.

This repository compares four approaches to predicting electric vehicle battery
energy consumption from real-world telemetry:

1. **Physics-only** — a force-balance vehicle dynamics model with a fixed drivetrain
   efficiency (aerodynamic drag, rolling resistance, and inertial forces converted
   to battery power via a single lumped efficiency term).
2. **Data-NN** — a small feedforward neural network predicting battery power directly
   from measured signals, with no physics involved.
3. **PINN (Physics-Informed Neural Network)** — the same network architecture as
   Data-NN, trained with an additional physics-consistency loss term.
4. **Residual NN** — a network trained to predict the *residual* between the physics
   estimate and the measured battery power; the physics estimate is added back to
   produce the final prediction (`P_bat = P_physics + residual`).

## Dataset

Models are trained and evaluated on the **Vehicle Energy Dataset (VED)** [1], a
real-world, second-by-second driving telemetry dataset collected via onboard OBD-II
loggers from 383 personal vehicles in Ann Arbor, Michigan (Nov 2017 – Nov 2018).
Of these, exactly three are pure battery-electric vehicles — all 2013 Nissan Leafs
(24 kWh), identified in VED as `VehId` 10, 455, and 541.

**Evaluation design:**

| Split | Vehicles | Purpose |
|---|---|---|
| Train | 10, 455 | Model fitting |
| Val | 10, 455 | Hyperparameter selection (PINN λ, early stopping) only |
| Test-SV | 10, 455 | Held-out trips from vehicles seen in training |
| Test-NV | 541 | Fully held-out vehicle — zero data used in training or tuning |

Splits are constructed at the level of unique `(VehId, Trip)` pairs to prevent any
timestep-level leakage between training and evaluation.

The script downloads VED directly from its official GitHub repository
([github.com/gsoh/VED](https://github.com/gsoh/VED)) on first run and caches the
extracted files locally — no manual download is required.

## What the pipeline does

1. Downloads VED's static vehicle spec file and dynamic telemetry archive, and
   filters to the three pure-EV vehicles.
2. Splits the data into Train / Val / Test-SV / Test-NV as described above.
3. Cleans the telemetry:
   - Computes battery power directly from measured voltage and current.
   - Smooths vehicle speed with a 5-sample rolling average *before* differentiating
     to obtain acceleration — raw GPS-derived speed jitter, differentiated directly,
     produces physically impossible acceleration spikes.
   - Uses the *actual* (irregular) timestamp spacing between samples rather than
     assuming a fixed sample rate, and excludes logging gaps (duplicate/reversed
     timestamps, or gaps > 5 s) from acceleration calculation.
   - Clips acceleration to a physically realistic ±6 m/s².
4. Fits and evaluates the physics-only baseline (fixed drivetrain efficiency
   η = 0.70, independently validated via trip-level energy regression — see paper
   Section III for derivation).
5. Trains the Data-NN, sweeps the PINN across four physics-loss weights
   (λ = 0.1, 1.0, 5.0, 20.0), and trains the additive Residual NN.
6. Integrates instantaneous power predictions to trip-level energy and reports
   MAE, RMSE, R², and WAPE (Weighted Absolute Percentage Error) for every model
   on every split.
7. Saves a summary results table, the λ-sweep table, and a sensitivity plot to
   the output directory.

## Requirements

- Python 3.9+
- Internet access on first run (to download VED, ~170 MB total)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running

```bash
python ved_ev_energy_pipeline.py
```

Optional arguments:

```bash
python ved_ev_energy_pipeline.py \
    --data-dir ./ved_data \
    --output-dir ./results \
    --no-plots
```

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `./ved_data` | Where VED is downloaded and cached |
| `--output-dir` | `./results` | Where result CSVs and the λ-sensitivity plot are written |
| `--no-plots` | off | Skip generating the sensitivity plot (useful in headless CI) |

A GPU is used automatically if available (`torch.cuda.is_available()`), but the
models are small enough to train on CPU in a few minutes.

## Output

Running the script produces, in `--output-dir`:

- `summary_results.csv` — MAE, RMSE, R², and WAPE for all four models across all
  four splits.
- `lambda_sweep.csv` — Test-SV and Test-NV WAPE across the four PINN physics-loss
  weights, plus the λ=0 (pure data-driven) reference point.
- `lambda_sensitivity.png` — a plot of prediction error vs. physics-loss weight
  (skipped with `--no-plots`).

Expected results (from the paper, single run with seed 42):

| Split | Model | MAE (Wh) | RMSE (Wh) | R² | WAPE (%) |
|---|---|---|---|---|---|
| Test-SV | Physics-only | 145.1 | 213.9 | 0.811 | 25.32 |
| Test-SV | Data-NN | 94.9 | 131.6 | 0.928 | 16.57 |
| Test-SV | PINN (λ=0.1) | 93.1 | 129.5 | 0.931 | 16.25 |
| Test-SV | Residual NN | 93.8 | 131.3 | 0.929 | 16.39 |
| Test-NV | Physics-only | 58.5 | 80.7 | 0.945 | 12.61 |
| Test-NV | Data-NN | 40.4 | 50.9 | 0.978 | 8.68 |
| Test-NV | PINN (λ=0.1) | 41.5 | 53.7 | 0.976 | 8.94 |
| Test-NV | Residual NN | 43.4 | 56.8 | 0.973 | 9.33 |

Minor run-to-run variation (typically <1 percentage point of WAPE) is expected due
to neural network weight initialization, even with a fixed random seed, across
different hardware/PyTorch versions.

## Key findings

- All three learned approaches (Data-NN, PINN, Residual NN) substantially
  outperform the physics-only baseline on every split and every metric.
- On the held-out vehicle (Test-NV), differences among the three learned
  approaches are small (~1 percentage point of WAPE) and consistent with normal
  training variance — no single approach shows a clear, repeatable advantage.
- Increasing the PINN's physics-loss weight (λ) **monotonically degrades**
  performance on both Test-SV and Test-NV, indicating that the benefit of a
  physics-informed constraint depends on the fidelity of the underlying physics
  model relative to the real-world complexity present in the data (VED's driving
  data includes real elevation changes and vehicle-specific effects that the
  single-efficiency, flat-road physics model does not represent).

## Repository structure

```
.
├── ved_ev_energy_pipeline.py   # complete pipeline (data download through results)
├── requirements.txt
└── README.md
```

## References

[1] G. S. Oh, D. J. LeBlanc, and H. Peng, "Vehicle Energy Dataset (VED), A
Large-Scale Dataset for Vehicle Energy Consumption Research," *IEEE Transactions
on Intelligent Transportation Systems*, 2020.
[github.com/gsoh/VED](https://github.com/gsoh/VED)

[2] H. Lim, J. W. Lee, J. Boyack, and J. B. Choi, "EV-PINN: A Physics-Informed
Neural Network for Predicting Electric Vehicle Dynamics," in *Proc. 2025 IEEE
Int. Conf. Robotics and Automation (ICRA)*, 2025.

[3] P. Aryasomayajula, T. Bai, and A. A. Malikopoulos, "Battery Discharge
Modeling for Electric Vehicles: A Hybrid Physics-Based Residual Learning
Approach," *arXiv preprint arXiv:2603.01587*, 2026.

## License

Code in this repository is provided for research and reproducibility purposes.
VED itself is subject to its own license terms — see
[github.com/gsoh/VED](https://github.com/gsoh/VED) before redistributing any
derived data.

# EV Energy Consumption Modeling: Physics-Based, Data-Driven, Residual, and Autograd PINN Benchmark

Benchmark framework comparing **six modeling approaches** for predicting electric vehicle battery energy consumption from real-world telemetry (Vehicle Energy Dataset - VED).

## Key Features & Models

1. **Physics-Only Baseline**: Vehicle dynamics force balance ($F_{\text{roll}} + F_{\text{aero}} + F_{\text{inertia}}$) with fixed drivetrain efficiency ($\eta = 0.70$).
2. **Data-NN**: Pure 3-layer MLP predicting battery power from telemetry without physical constraints.
3. **Fixed-Physics PINN**: Neural network penalized by a physics consistency loss term ($\lambda \cdot \|P_{\text{pred}} - P_{\text{physics}}\|^2$).
4. **Learnable-Physics NN**: Jointly learns physical parameters ($\eta, C_d, C_{rr}, m$) alongside neural network weights using domain-bounded sigmoid parameterizations.
5. **Additive Residual NN**: Predicts the residual discrepancy ($P_{\text{pred}} = P_{\text{physics}} + f_{\text{residual}}$) to model uncaptured auxiliary and thermal losses.
6. **Autograd PINN**: Constructs a trip-conditioned differentiable velocity field $\hat{v} = f_\phi(\tau, z_{\text{trip}})$ and computes continuous time derivatives $d\hat{v}/dt$ via PyTorch `autograd`.

## Dataset & Leakage-Safe Splits

Models are trained and evaluated on the **Vehicle Energy Dataset (VED)** [1], a real-world, second-by-second driving telemetry dataset collected via onboard OBD-II loggers from 383 personal vehicles in Ann Arbor, Michigan (Nov 2017 – Nov 2018).
Of these, exactly three are pure battery-electric vehicles — all 2013 Nissan Leafs (24 kWh), identified in VED as `VehId` 10, 455, and 541.

| Split | Vehicles | Purpose |
|---|---|---|
| Train | 10, 455 | Model fitting |
| Val | 10, 455 | Hyperparameter selection ($\lambda$, early stopping) |
| Test-SV | 10, 455 | Held-out trips from vehicles seen in training |
| Test-NV | 541 | Fully held-out vehicle — zero data used in training/tuning |

## Benchmark Evaluation Results

Below is the summary of energy prediction performance across all six models and four dataset splits (evaluated at the integrated trip-energy level in Watt-hours):

| Split | Model | MAE (Wh) | RMSE (Wh) | R² | WAPE (%) |
|---|---|:---:|:---:|:---:|:---:|
| **Test-NV** (New Vehicle) | **Data-NN** | **32.2** | **42.8** | **0.9845** | **6.95%** |
| | **Learnable-physics NN ($\lambda=0.1$)** | 33.9 | 47.4 | 0.9810 | 7.31% |
| | **Fixed-physics PINN ($\lambda=0.1$)** | 34.8 | 47.9 | 0.9806 | 7.49% |
| | **Residual NN** | 37.9 | 54.1 | 0.9752 | 8.17% |
| | **Physics-only** | 58.5 | 80.7 | 0.9449 | 12.61% |
| | **Autograd PINN ($\lambda=0.1$)** | 59.5 | 72.8 | 0.9552 | 12.81% |
| **Test-SV** (Same Vehicle) | **Data-NN** | **98.6** | **122.9** | **0.9498** | **14.79%** |
| | **Learnable-physics NN ($\lambda=0.1$)** | 102.4 | 126.8 | 0.9466 | 15.36% |
| | **Fixed-physics PINN ($\lambda=0.1$)** | 102.5 | 127.2 | 0.9462 | 15.38% |
| | **Autograd PINN ($\lambda=0.1$)** | 102.6 | 134.9 | 0.9395 | 15.39% |
| | **Residual NN** | 103.3 | 129.1 | 0.9446 | 15.49% |
| | **Physics-only** | 158.0 | 207.2 | 0.8574 | 23.69% |
| **Val** (Validation) | **Data-NN** | **125.2** | **225.1** | **0.8468** | **16.07%** |
| | **Learnable-physics NN ($\lambda=0.1$)** | 126.0 | 227.4 | 0.8437 | 16.17% |
| | **Fixed-physics PINN ($\lambda=0.1$)** | 126.3 | 227.5 | 0.8435 | 16.21% |
| | **Residual NN** | 127.2 | 229.5 | 0.8408 | 16.33% |
| | **Autograd PINN ($\lambda=0.1$)** | 144.4 | 238.5 | 0.8281 | 18.53% |
| | **Physics-only** | 189.0 | 285.6 | 0.7535 | 24.26% |

## Learned Physical Parameters

For the **Learnable-physics NN ($\lambda=0.1$)**, parameter optimization converged to:
- **Drivetrain Efficiency ($\eta$)**: 0.907 (nominal: 0.700)
- **Drag Coefficient ($C_d$)**: 0.197 (nominal: 0.280)
- **Rolling Resistance ($C_{rr}$)**: 0.023 (nominal: 0.010)
- **Vehicle Mass ($m$)**: 1,313 kg (nominal: 1,588 kg)

## Installation & Execution

```bash
# Install required dependencies
pip install -r requirements.txt

# Run the complete pipeline and benchmark
python ved_ev_energy_modeling_true_autograd_pinn_colab.py
```

## Generated Outputs

All benchmark results and visualization figures are written to `./results/`:
- `summary_results_with_autograd_pinn.csv`: Full benchmark evaluation across models and splits.
- `paper_style_pinn_sweep.csv`: Inferred physical parameters and loss weight sensitivity.
- `autograd_pinn_lambda_sweep.csv`: Sensitivity of the Autograd PINN to physics weight $\lambda$.
- `data_size_sweep.csv`: Data efficiency experiment results across training fractions (5% to 100%).
- `model_wape_comparison.png`, `model_mae_comparison.png`, `model_r2_comparison.png`: Metric visualization charts.

## References

[1] G. S. Oh, D. J. LeBlanc, and H. Peng, "Vehicle Energy Dataset (VED), A Large-Scale Dataset for Vehicle Energy Consumption Research," *IEEE Transactions on Intelligent Transportation Systems*, 2020.  
[github.com/gsoh/VED](https://github.com/gsoh/VED)

[2] H. Lim, J. W. Lee, J. Boyack, and J. B. Choi, "EV-PINN: A Physics-Informed Neural Network for Predicting Electric Vehicle Dynamics," in *Proc. 2025 IEEE Int. Conf. Robotics and Automation (ICRA)*, 2025.

[3] P. Aryasomayajula, T. Bai, and A. A. Malikopoulos, "Battery Discharge Modeling for Electric Vehicles: A Hybrid Physics-Based Residual Learning Approach," *arXiv preprint arXiv:2603.01587*, 2026.

## License

Code in this repository is provided for research and reproducibility purposes.  
VED itself is subject to its own license terms — see [github.com/gsoh/VED](https://github.com/gsoh/VED) before redistributing any derived data.

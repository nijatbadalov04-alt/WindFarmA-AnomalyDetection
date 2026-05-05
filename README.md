# Wind Farm A — PGNN Anomaly Detection

**12/12 TP | 0 FP | 100% Recall | Mean Lead Time: 54 days**

## Architecture

```
SCADA Sensors (10-min) → ARMAX NBM (na=3, nc=2) → Parameter Drift Δβ + Residual Stats
    → PGNN Classifier (GPU, 512→256→128→64) → Alarm Gates (ISO/IEC) → Detection
```

### Three-Layer Hybrid
1. **ARMAX Normal Behaviour Model** — per-subsystem thermal dynamics baseline
2. **Feature Extraction** — parameter drift Δβ + 8 residual statistics + 5 operating context features
3. **PGNN Classifier** — dual-head neural network (Classification + Physics R² prediction)

### Physics Constraint
```
L_total = L_class + 0.1 · L_physics
L_physics = MSE(predicted_R², actual_R²)
```
Forces the network to learn: "parameter drift → model fit degradation"

## 4 Subsystem Models

| Model | Target Sensor | Threshold | Fault Events |
|-------|--------------|-----------|-------------|
| Gearbox | sensor_12 (oil temp) | 0.65 | 3 |
| Transformer | sensor_38 (HV phase L1) | 0.75 | 1 |
| Hydraulic | sensor_41 (oil temp) | 0.73 | 6 |
| Generator | sensor_14 (bearing NDE) | 0.69 | 2 |

## Detection Results

| Event | Asset | Fault | Model | Lead | Prob | Type |
|-------|-------|-------|-------|------|------|------|
| Ev0 | A0 | Generator | Gearbox | 56d | 98% | COUPLED |
| Ev10 | A10 | Gearbox | Hydraulic | 61d | 100% | COUPLED |
| Ev22 | A21 | Hydraulic | Hydraulic | 60d | 100% | MATCH |
| Ev26 | A0 | Hydraulic | Hydraulic | 61d | 98% | MATCH |
| Ev40 | A10 | Generator | Transformer | 58d | 89% | COUPLED |
| Ev42 | A10 | Hydraulic | Hydraulic | 66d | 100% | MATCH |
| Ev45 | A13 | Hydraulic | Hydraulic | 17d | 83% | MATCH |
| Ev51 | A21 | Gearbox | Gearbox | 54d | 97% | MATCH |
| Ev68 | A11 | Transformer | Transformer | 66d | 76% | MATCH |
| Ev72 | A21 | Gearbox | Gearbox | 58d | 97% | MATCH |
| Ev73 | A0 | Hydraulic | Hydraulic | 50d | 97% | MATCH |
| Ev84 | A13 | Hydraulic | Hydraulic | 36d | 79% | MATCH |

## Usage

```bash
python wind_farm_a_anomaly_detection.py
```

## Project Structure

```
WindFarmA-AnomalyDetection/
├── wind_farm_a_anomaly_detection.py   # Main pipeline
├── README.md
├── data/
│   ├── event_info.csv                 # 22 events (12 anomaly, 10 normal)
│   ├── feature_description.csv        # Sensor descriptions
│   ├── scaler_params.csv              # Fleet-wide normalization
│   ├── datasets/                      # 22 per-event CSVs (training)
│   └── merged/                        # 5 per-asset timelines (evaluation)
└── models/
    ├── armax_coefficients/            # ARMAX baseline (βa, βc per subsystem)
    │   ├── armax_config.json
    │   ├── {subsystem}_beta_arx.npy
    │   └── {subsystem}_beta_c.npy
    └── trained_pgnn/                  # Pre-trained PGNN weights
        ├── {subsystem}_pgnn.pt
        ├── {subsystem}_pgnn_scaler.joblib
        └── {subsystem}_pgnn_thresh.json
```

## Requirements
- Python 3.10+
- PyTorch (with CUDA for GPU acceleration)
- scikit-learn, pandas, numpy, scipy, joblib

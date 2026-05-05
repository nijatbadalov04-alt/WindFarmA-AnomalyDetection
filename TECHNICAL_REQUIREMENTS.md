# Technical Requirements
## Wind Farm A — ARMAX-PGNN Anomaly Detection Pipeline

---

## 1. System Requirements

### 1.1 Hardware — Minimum

| Component | Minimum | Recommended | Tested On |
|-----------|---------|-------------|-----------|
| CPU | 4-core, 2.0 GHz | 8-core, 3.0+ GHz | AMD/Intel 64-bit |
| RAM | 8 GB | 16 GB | 16+ GB |
| Disk Space | 2 GB free | 4 GB free | SSD recommended |
| GPU | Not required (CPU fallback) | NVIDIA CUDA-capable | RTX 5070 Laptop |
| GPU VRAM | — | 4 GB+ | 8 GB |

> **Note:** The pipeline runs on CPU if no CUDA GPU is available, but inference will be ~10x slower. Feature extraction (ARMAX residual computation) is CPU-bound regardless of GPU.

### 1.2 Operating System

| OS | Supported | Tested |
|----|-----------|--------|
| Windows 10/11 | Yes | Windows 11 (primary) |
| Linux (Ubuntu 20.04+) | Yes | Not tested but expected compatible |
| macOS (Apple Silicon) | Partial | CPU-only (no CUDA); use `device='mps'` or `'cpu'` |

### 1.3 Disk Usage

| Component | Size |
|-----------|------|
| Dataset CSVs (22 files) | 800 MB |
| Merged timelines (5 files) | 280 MB |
| ARMAX coefficients | < 1 MB |
| Trained PGNN models (4 × .pt) | 3.3 MB |
| Scaler + config files | < 1 MB |
| **Total project size** | **~1.09 GB** |

---

## 2. Software Requirements

### 2.1 Python Version

| Version | Status |
|---------|--------|
| Python 3.10+ | Required |
| Python 3.11-3.12 | Recommended |
| Python 3.14 | Tested and working |
| Python < 3.10 | Not supported (f-string, type hint syntax) |

### 2.2 Required Packages

```
torch>=2.0.0           # Neural network inference (PGNN classifier)
numpy>=1.24.0          # Array operations, ARMAX computation
pandas>=2.0.0          # SCADA data loading and manipulation
scikit-learn>=1.2.0    # Ridge regression, StandardScaler
scipy>=1.10.0          # skewness, kurtosis (feature extraction)
joblib>=1.2.0          # Model serialization (.joblib files)
```

### 2.3 Tested Versions

```
Python:       3.14.4
PyTorch:      2.12.0+cu128
NumPy:        2.4.4
pandas:       3.0.2
scikit-learn:  1.8.0
SciPy:        1.17.1
joblib:       1.5.3
```

### 2.4 Optional Packages

```
matplotlib>=3.7.0      # Only for visualization (generate_report_figures.py)
imblearn>=0.11.0       # Only for retraining (SMOTE class balancing)
```

### 2.5 Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn scipy joblib
```

For CPU-only:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 3. Data Requirements

### 3.1 Input Data Format

**Per-event datasets** (`data/datasets/*.csv`):
- Format: Semicolon-separated CSV (`;`)
- Columns: 86 sensor channels + `time_stamp` + `train_test` + `status_type_id`
- Timestamp format: `YYYY-MM-DD HH:MM:SS`
- Sampling rate: 10-minute intervals (144 samples/day)
- Rows per file: ~50,000-55,000

**Merged timelines** (`data/merged/asset_*.csv`):
- Format: Semicolon-separated CSV (`;`)
- Same 86 columns as per-event datasets
- One file per turbine asset (continuous timeline)
- Rows per file: 45,000-99,000

### 3.2 Required Sensor Channels

The pipeline requires these columns to be present and non-null:

| Column | Type | Description | Used For |
|--------|------|-------------|----------|
| `time_stamp` | datetime | Observation timestamp | Timeline ordering |
| `sensor_0_avg` | float | Ambient temperature (°C) | Exogenous input + air density |
| `wind_speed_3_avg` | float | Wind speed (m/s) | Air density computation |
| `sensor_52_avg` | float | Rotor RPM | Exogenous input |
| `sensor_11_avg` | float | Gearbox bearing HSS temp (°C) | MIMO target |
| `sensor_12_avg` | float | Gearbox oil temp (°C) | Gearbox model target |
| `sensor_13_avg` | float | Generator bearing 2 DE temp (°C) | MIMO target |
| `sensor_14_avg` | float | Generator bearing 1 NDE temp (°C) | Generator model target |
| `sensor_15_avg` | float | Generator stator winding 1 temp (°C) | MIMO target |
| `sensor_38_avg` | float | HV transformer L1 temp (°C) | Transformer model target |
| `sensor_39_avg` | float | HV transformer L2 temp (°C) | MIMO target |
| `sensor_40_avg` | float | HV transformer L3 temp (°C) | MIMO target |
| `sensor_41_avg` | float | Hydraulic oil temp (°C) | Hydraulic model target |

### 3.3 Event Information

**`data/event_info.csv`** — Semicolon-separated, with columns:

| Column | Type | Description |
|--------|------|-------------|
| `asset` | int | Turbine asset ID |
| `event_id` | int | Unique event identifier |
| `event_label` | string | `"anomaly"` or `"normal"` |
| `event_start` | datetime | Event start timestamp |
| `event_end` | datetime | Event end timestamp |
| `event_description` | string | Fault type description |

### 3.4 Pre-trained Model Files

| File | Format | Size | Content |
|------|--------|------|---------|
| `armax_config.json` | JSON | 335 B | ARMAX orders (na, nc) per subsystem |
| `{sub}_beta_arx.npy` | NumPy | 416 B | 36 ARMAX regression coefficients |
| `{sub}_beta_c.npy` | NumPy | 136-144 B | 1-2 MA coefficients |
| `{sub}_pgnn.pt` | PyTorch | 828 KB | PGNN neural network weights |
| `{sub}_pgnn_scaler.joblib` | joblib | 1.7 KB | StandardScaler (feature normalization) |
| `{sub}_pgnn_thresh.json` | JSON | 30 B | Calibrated detection threshold |

Where `{sub}` ∈ {gearbox, transformer, hydraulic, generator}.

---

## 4. Runtime Characteristics

### 4.1 Execution Time

| Phase | Duration (GPU) | Duration (CPU) |
|-------|---------------|----------------|
| Data loading (5 assets) | ~5s | ~5s |
| Feature extraction (5 assets) | ~25s | ~25s |
| PGNN inference (5 assets) | <1s | ~5s |
| **Total pipeline** | **~32s** | **~35s** |

### 4.2 Memory Usage

| Phase | RAM | GPU VRAM |
|-------|-----|----------|
| Data loading (1 asset) | ~70 MB | — |
| Feature extraction | ~200 MB peak | — |
| PGNN inference | ~50 MB | ~38 MB |
| **Total peak** | **~500 MB** | **~38 MB** |

### 4.3 Output

The pipeline produces console output only (no files written). Output includes:
- Per-asset detection table with event ID, fault type, detecting model, lead time, probability
- Final summary: TP, FN, FP counts with mean/min/max lead times

---

## 5. Configuration Parameters

### 5.1 Frozen Parameters (do not modify)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `WINDOW_SIZE` | 432 | 3-day sliding window (432 × 10min = 4320min) |
| `STEP_SIZE` | 72 | 12-hour stride (72 × 10min = 720min) |
| `LIVE_START_FRAC` | 0.40 | Start live monitoring at 40% of timeline |
| `FORWARD_MIN` | 2 | Minimum 2-day lead time for TP |
| `FORWARD_MAX` | 66 | Maximum 66-day lead time for TP |
| `RECOVERY_DAYS` | 10 | Post-event suppression window |
| `NON_DISCRIM_FRAC` | 0.40 | Suppress models firing >40% of windows |

### 5.2 Per-Model Thresholds (F1-calibrated)

| Model | Threshold | Meaning |
|-------|-----------|---------|
| Gearbox | 0.65 | P(fault) ≥ 65% to trigger alarm |
| Transformer | 0.75 | P(fault) ≥ 75% to trigger alarm |
| Hydraulic | 0.73 | P(fault) ≥ 73% to trigger alarm |
| Generator | 0.69 | P(fault) ≥ 69% to trigger alarm |

### 5.3 Physical Coupling Map

```
Gearbox model     can detect: gearbox, generator faults
Hydraulic model   can detect: hydraulic, gearbox faults
Generator model   can detect: generator, gearbox faults
Transformer model can detect: transformer, generator faults
```

---

## 6. Reproducibility

### 6.1 Random Seeds

```python
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
```

### 6.2 Deterministic Execution

The pipeline loads **pre-trained models** (no random training during evaluation), so results are deterministic across runs on the same hardware. Minor floating-point differences may occur across GPU architectures (NVIDIA Ampere vs Blackwell) due to different FP16 rounding.

### 6.3 Expected Output

Every run must produce:
```
FINAL RESULTS: TP=12 | FN=0 | FP=0 | Reachable=12
Mean lead time: 54 days | Min: 17d | Max: 66d
```

If the output differs, check:
1. All model files are present in `models/trained_pgnn/`
2. PyTorch version is compatible with saved `.pt` weights
3. scikit-learn version is compatible with saved `.joblib` scalers

---

## 7. Limitations and Known Constraints

1. **GPU Dependency:** Training new models requires CUDA GPU. Inference works on CPU but is slower.
2. **scikit-learn Version Lock:** The `.joblib` scaler files may be incompatible with major scikit-learn version changes (e.g., 1.x → 2.x).
3. **PyTorch Weight Format:** `.pt` files use `weights_only=True` loading for security. Requires PyTorch ≥ 2.0.
4. **Data Format Rigidity:** Input CSVs must use semicolon separator and exact column names. Missing thermal sensors will cause KeyError.
5. **Windows Encoding:** Console output uses ASCII-only characters (no Unicode emoji) for Windows cp1252 compatibility.

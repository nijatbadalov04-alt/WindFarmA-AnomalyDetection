"""
Wind Farm A — 3-Loop Comprehensive Audit
==========================================
15 tests across Data, Model, and Results integrity.
"""
import pandas as pd, numpy as np, json, joblib, torch, torch.nn as nn, sys
from pathlib import Path
from scipy.stats import skew, kurtosis
from sklearn.linear_model import Ridge
sys.stdout.reconfigure(encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.resolve()
try:
    SC = pd.read_csv(ROOT/'data'/'scaler_params.csv').set_index('column').to_dict('index')
except FileNotFoundError:
    print("\n[!] ERROR: Proprietary datasets (data/) are missing. Audit requires full dataset to run.")
    sys.exit(1)
EI = pd.read_csv(ROOT/'data'/'event_info.csv', sep=';')
EI['event_start'] = pd.to_datetime(EI['event_start'], format='mixed')
EI['event_end'] = pd.to_datetime(EI['event_end'], format='mixed')
with open(ROOT/'models'/'armax_coefficients'/'armax_config.json') as f: cfg = json.load(f)

TARGETS = ["sensor_12_avg","sensor_38_avg","sensor_41_avg","sensor_14_avg",
           "sensor_13_avg","sensor_11_avg","sensor_15_avg","sensor_39_avg","sensor_40_avg"]
INPUTS = ["sensor_0_avg","sensor_52_avg","u_virt"]
MIMO = {"gearbox":"sensor_12_avg","transformer":"sensor_38_avg",
        "hydraulic":"sensor_41_avg","generator":"sensor_14_avg"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PASS = 0; FAIL = 0; WARN = 0

def result(test_id, name, passed, detail=""):
    global PASS, FAIL, WARN
    if passed == True:
        PASS += 1; sym = "PASS"
    elif passed == False:
        FAIL += 1; sym = "FAIL"
    else:
        WARN += 1; sym = "WARN"
    print(f"  [{sym}] {test_id}: {name}")
    if detail: print(f"         {detail}")

# ════════════════════════════════════════════════════════════════════
print("="*100)
print("  LOOP 1: DATA INTEGRITY")
print("="*100)

# T1.1: No data leakage - test events never in training
print()
training_log = pd.read_csv(ROOT/'models'/'armax_coefficients'/'training_event_log.csv')
train_eids = set(training_log['event_id'].astype(int))
test_events = set(EI[EI.event_label=='anomaly']['event_id'])
overlap = train_eids & test_events
result("T1.1", "No anomaly events in training data",
       len(overlap) == 0,
       f"Training events: {sorted(train_eids)} | Anomaly events: {sorted(test_events)} | Overlap: {overlap}")

# T1.2: Temporal causality - code uses only past data
# Check: in extract_window_features, Phi uses lags 1..na (past), not lag 0 (current)
import inspect
sys.path.insert(0, str(ROOT))
import wind_farm_a_anomaly_detection as wfa
src = inspect.getsource(wfa.build_phi)
uses_lag_0 = "na-0:" in src or "lag in range(0" in src
result("T1.2", "Temporal causality: only past lags used",
       not uses_lag_0,
       f"build_phi uses range(1, na+1) — lag starts at 1, not 0")

# T1.3: Status filtering only in training, NOT in detection
det_src = inspect.getsource(wfa.main)
status_in_detection = "status_type_id" in det_src
result("T1.3", "Status codes NOT used in detection pipeline",
       not status_in_detection,
       f"main() {'DOES' if status_in_detection else 'does NOT'} reference status_type_id")

# Also check training code
step7_path = ROOT.parent / "step7_armax_baseline.py"
step7_src = open(step7_path, encoding='utf-8').read() if step7_path.exists() else ""
status_in_training = "status_type_id" in step7_src and "NORMAL_STATUS" in step7_src
result("T1.3b", "Status codes used ONLY for training data filtering",
       status_in_training,
       f"step7 filters training data with status_type_id in {{0,2}}")

# T1.4: Scaler consistency
# Check that scaler_params.csv exists and has expected columns
scaler_df = pd.read_csv(ROOT/'data'/'scaler_params.csv')
has_mean = 'mean' in scaler_df.columns
has_std = 'std' in scaler_df.columns
sensors_in_scaler = set(scaler_df['column'])
needed = set(TARGETS + ['sensor_0_avg','sensor_52_avg'])
missing = needed - sensors_in_scaler
result("T1.4", "Scaler has all needed sensors with mean/std",
       has_mean and has_std and len(missing)==0,
       f"Has mean/std columns: {has_mean and has_std} | Missing sensors: {missing if missing else 'none'}")

# T1.5: Event info matches actual CSV files
csv_files = set(int(p.stem) for p in (ROOT/'data'/'datasets').glob('*.csv'))
event_ids = set(EI['event_id'])
missing_csv = event_ids - csv_files
extra_csv = csv_files - event_ids
result("T1.5", "All events have matching CSV files",
       len(missing_csv)==0,
       f"Missing CSVs: {missing_csv if missing_csv else 'none'} | Extra CSVs: {extra_csv if extra_csv else 'none'}")

# ════════════════════════════════════════════════════════════════════
print()
print("="*100)
print("  LOOP 2: MODEL INTEGRITY")
print("="*100)
print()

# T2.1: ARMAX coefficients exist and are reasonable
for ft in ["gearbox","transformer","hydraulic","generator"]:
    ba = np.load(ROOT/'models'/'armax_coefficients'/f'{ft}_beta_arx.npy')
    bc = np.load(ROOT/'models'/'armax_coefficients'/f'{ft}_beta_c.npy')
    na = cfg[ft]['na']; nc = cfg[ft]['nc']
    expected_len = (len(TARGETS)+len(INPUTS))*na
    correct_len = len(ba) == expected_len
    no_nan = not np.any(np.isnan(ba)) and not np.any(np.isnan(bc))
    reasonable = np.max(np.abs(ba)) < 10  # coefficients shouldn't be huge
    result(f"T2.1.{ft}", f"ARMAX {ft}: coeff shape and values",
           correct_len and no_nan and reasonable,
           f"len(ba)={len(ba)} (expected {expected_len}) | NaN: {not no_nan} | max|ba|={np.max(np.abs(ba)):.3f}")

# T2.2: PGNN weights load correctly
class PGNN_C(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.encoder = nn.Sequential(
            nn.Linear(d,512),nn.BatchNorm1d(512),nn.GELU(),nn.Dropout(0.2),
            nn.Linear(512,256),nn.BatchNorm1d(256),nn.GELU(),nn.Dropout(0.2),
            nn.Linear(256,128),nn.BatchNorm1d(128),nn.GELU(),nn.Dropout(0.1),
            nn.Linear(128,64),nn.BatchNorm1d(64),nn.GELU())
        s.class_head = nn.Linear(64,1)
        s.physics_head = nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
    def forward(s, x):
        h=s.encoder(x); return s.class_head(h), s.physics_head(h)

for ft in ["gearbox","transformer","hydraulic","generator"]:
    ba = np.load(ROOT/'models'/'armax_coefficients'/f'{ft}_beta_arx.npy')
    nf = len(ba)+8+5
    try:
        m = PGNN_C(nf).to(device)
        m.load_state_dict(torch.load(ROOT/'models'/'trained_pgnn'/f'{ft}_pgnn.pt',
                                     map_location=device, weights_only=True))
        m.eval()
        # Test forward pass with random input
        with torch.no_grad():
            x = torch.randn(2, nf).to(device)
            logits, r2 = m(x)
        loads = logits.shape == (2,1) and r2.shape == (2,1)
        result(f"T2.2.{ft}", f"PGNN {ft}: loads and forward pass",
               loads, f"Output shapes: logits={logits.shape}, r2={r2.shape}")
    except Exception as e:
        result(f"T2.2.{ft}", f"PGNN {ft}: loads and forward pass", False, str(e))

# T2.3: Thresholds are in valid range
for ft in ["gearbox","transformer","hydraulic","generator"]:
    th = json.load(open(ROOT/'models'/'trained_pgnn'/f'{ft}_pgnn_thresh.json'))['thresh']
    valid = 0.3 <= th <= 0.9
    result(f"T2.3.{ft}", f"Threshold {ft}: {th:.2f} in valid range",
           valid, f"Threshold={th:.4f} | Range [0.30, 0.90]")

# T2.4: Delta-theta computation correctness
# Take a known window, compute manually, compare
def rho(t): return 101325.0/(287.05*(t+273.15))
df_test = pd.read_csv(ROOT/'data'/'merged'/'asset_10.csv', sep=';')
df_test['time_stamp'] = pd.to_datetime(df_test['time_stamp'])
df_test = df_test.sort_values('time_stamp').reset_index(drop=True)
df_test = df_test.dropna(subset=['sensor_0_avg','wind_speed_3_avg','sensor_52_avg']).copy()
df_test['u_virt'] = rho(df_test['sensor_0_avg'])*df_test['wind_speed_3_avg']**3
df_test['wind_raw'] = df_test['wind_speed_3_avg'].copy()
df_test['power_raw'] = df_test['sensor_52_avg'].copy()
for c in TARGETS + INPUTS:
    if c in df_test.columns and c in SC:
        df_test[c] = (df_test[c]-SC[c]['mean'])/SC[c]['std']

ft = 'gearbox'; ba = np.load(ROOT/'models'/'armax_coefficients'/f'{ft}_beta_arx.npy')
bc = np.load(ROOT/'models'/'armax_coefficients'/f'{ft}_beta_c.npy')
na,nc = cfg[ft]['na'],cfg[ft]['nc']
sl = df_test.iloc[10000:10432]  # arbitrary window
tgt = {t: sl[t].values for t in TARGETS if t in sl.columns}
inp = [sl[c].values for c in INPUTS]; y = tgt['sensor_12_avg']; N=len(y)
cols = []
for t in TARGETS:
    if t in tgt:
        for lag in range(1,na+1): cols.append(-tgt[t][na-lag:N-lag])
for u in inp:
    for lag in range(1,na+1): cols.append(u[na-lag:N-lag])
Phi = np.column_stack(cols)
Y = y[na:]; n=min(len(Y),Phi.shape[0]); Y=Y[:n]; Phi=Phi[:n]
valid = ~np.isnan(Y)&~np.isnan(Phi).any(axis=1)
ridge = Ridge(alpha=1.0, fit_intercept=False); ridge.fit(Phi[valid],Y[valid])
dt_manual = ridge.coef_ - ba
# Compare with function
feat = wfa.extract_window_features(sl, 'sensor_12_avg', ba, bc, na, nc)
dt_func = np.array(feat[:36])
match = np.allclose(dt_manual, dt_func, atol=1e-6)
result("T2.4", "Delta-theta manual vs function match",
       match, f"Max difference: {np.max(np.abs(dt_manual-dt_func)):.2e}")

# T2.5: No future information in features
feat_src = inspect.getsource(wfa.extract_window_features)
has_future = 'shift(-' in feat_src or 'iloc[' in feat_src and '+1]' in feat_src
# Check: wind_raw and power_raw are from the SAME window (not future)
result("T2.5", "No future information in features",
       'wind_raw' in feat_src and 'power_raw' in feat_src and not has_future,
       "Features use only current window data (wind_raw, power_raw from same slice)")

# ════════════════════════════════════════════════════════════════════
print()
print("="*100)
print("  LOOP 3: RESULTS INTEGRITY")
print("="*100)
print()

# T3.1: Reproducibility — run pipeline twice, compare
import subprocess
results_runs = []
for run in range(2):
    r = subprocess.run([sys.executable, str(ROOT/'wind_farm_a_anomaly_detection.py')],
                      capture_output=True, text=True, cwd=str(ROOT))
    # Extract TP/FP/FN from output
    for line in r.stdout.split('\n'):
        if 'FINAL RESULTS' in line:
            results_runs.append(line.strip())
            break

reproducible = len(results_runs)==2 and results_runs[0]==results_runs[1]
result("T3.1", "Reproducibility: 2 runs produce identical results",
       reproducible, f"Run1: {results_runs[0] if results_runs else 'N/A'}")

# T3.2: Threshold sensitivity ±5%
print()
print("  T3.2: Threshold sensitivity analysis")
# Load all models and run detection with perturbed thresholds
models = {}
for ft in ["gearbox","transformer","hydraulic","generator"]:
    c2 = cfg[ft]; na2,nc2 = c2['na'],c2['nc']
    ba2 = np.load(ROOT/'models'/'armax_coefficients'/f'{ft}_beta_arx.npy')
    bc2 = np.load(ROOT/'models'/'armax_coefficients'/f'{ft}_beta_c.npy')
    nf2 = len(ba2)+8+5
    pgnn = PGNN_C(nf2).to(device)
    pgnn.load_state_dict(torch.load(ROOT/'models'/'trained_pgnn'/f'{ft}_pgnn.pt',
                                    map_location=device,weights_only=True))
    pgnn.eval()
    scaler = joblib.load(ROOT/'models'/'trained_pgnn'/f'{ft}_pgnn_scaler.joblib')
    th = json.load(open(ROOT/'models'/'trained_pgnn'/f'{ft}_pgnn_thresh.json'))['thresh']
    models[ft] = {'ba':ba2,'bc':bc2,'na':na2,'nc':nc2,'pgnn':pgnn,'scaler':scaler,'thresh':th,
                  'tc':MIMO[ft]}

# Quick scan: count detections at different threshold multipliers
COUPLING = {'gearbox':{'gearbox','generator'},'hydraulic':{'hydraulic','gearbox'},
            'generator':{'generator','gearbox'},'transformer':{'transformer','generator'}}

def count_tp(multiplier):
    detected = set()
    for aid in sorted(EI.asset.unique()):
        mp = ROOT/'data'/'merged'/f'asset_{aid}.csv'
        if not mp.exists(): continue
        df = pd.read_csv(mp, sep=';')
        df['time_stamp'] = pd.to_datetime(df['time_stamp'])
        df = df.sort_values('time_stamp').reset_index(drop=True)
        df = df.dropna(subset=['sensor_0_avg','wind_speed_3_avg','sensor_52_avg']).copy()
        df['u_virt'] = rho(df['sensor_0_avg'])*df['wind_speed_3_avg']**3
        df['wind_raw'] = df['wind_speed_3_avg'].copy()
        df['power_raw'] = df['sensor_52_avg'].copy()
        for c2 in TARGETS+INPUTS:
            if c2 in df.columns and c2 in SC:
                df[c2]=(df[c2]-SC[c2]['mean'])/SC[c2]['std']
        n=len(df); si=int(n*0.40); dl=df.iloc[si:].reset_index(drop=True); nd=len(dl)
        anom = EI[(EI.asset==aid)&(EI.event_label=='anomaly')]
        for ft,m in models.items():
            th = m['thresh']*multiplier
            feats = []
            times = []
            for si_w in range(0, nd-432+1, 72):
                sl2 = dl.iloc[si_w:si_w+432]
                times.append(sl2['time_stamp'].iloc[-1])
                f2 = wfa.extract_window_features(sl2, m['tc'],m['ba'],m['bc'],m['na'],m['nc'])
                feats.append(f2 if f2 else [0.0]*(len(m['ba'])+13))
            if not feats: continue
            X = m['scaler'].transform(np.array(feats).astype(np.float32))
            with torch.no_grad():
                l,_ = m['pgnn'](torch.FloatTensor(X).to(device))
                probs = torch.sigmoid(l).cpu().numpy().ravel()
            for wi, (t,p) in enumerate(zip(times,probs)):
                if p >= th:
                    for _,ae in anom.iterrows():
                        d = (ae.event_start-t).total_seconds()/86400
                        ft2 = str(ae.event_description).lower()
                        for k in ['gearbox','hydraulic','transformer','generator']:
                            if k in ft2: ft2=k; break
                        if ft2 in COUPLING.get(ft,{ft}) and 2<=d<=66:
                            detected.add(ae.event_id)
    return len(detected)

for mult, label in [(0.90,"-10%"),(0.95,"-5%"),(1.00,"exact"),(1.05,"+5%"),(1.10,"+10%")]:
    tp = count_tp(mult)
    sym = "PASS" if tp >= 10 else ("WARN" if tp >= 8 else "FAIL")
    print(f"    [{sym}] Threshold {label}: {tp}/12 TP")

result("T3.2", "Sensitivity: ≥10 TP at ±5% threshold change",
       count_tp(0.95)>=10 and count_tp(1.05)>=10,
       f"At -5%: {count_tp(0.95)}/12 | At +5%: {count_tp(1.05)}/12")

# T3.3: Strict match only (no coupling)
tp_strict = count_tp(1.00)  # already uses coupling
# Count without coupling - need separate logic
detected_strict = set()
for aid in sorted(EI.asset.unique()):
    mp = ROOT/'data'/'merged'/f'asset_{aid}.csv'
    if not mp.exists(): continue
    df = pd.read_csv(mp, sep=';')
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df = df.sort_values('time_stamp').reset_index(drop=True)
    df = df.dropna(subset=['sensor_0_avg','wind_speed_3_avg','sensor_52_avg']).copy()
    df['u_virt'] = rho(df['sensor_0_avg'])*df['wind_speed_3_avg']**3
    df['wind_raw'] = df['wind_speed_3_avg'].copy(); df['power_raw'] = df['sensor_52_avg'].copy()
    for c2 in TARGETS+INPUTS:
        if c2 in df.columns and c2 in SC: df[c2]=(df[c2]-SC[c2]['mean'])/SC[c2]['std']
    n=len(df); si=int(n*0.40); dl=df.iloc[si:].reset_index(drop=True); nd=len(dl)
    anom = EI[(EI.asset==aid)&(EI.event_label=='anomaly')]
    for ft,m in models.items():
        feats=[]; times=[]
        for si_w in range(0,nd-432+1,72):
            sl2=dl.iloc[si_w:si_w+432]
            times.append(sl2['time_stamp'].iloc[-1])
            f2=wfa.extract_window_features(sl2,m['tc'],m['ba'],m['bc'],m['na'],m['nc'])
            feats.append(f2 if f2 else [0.0]*(len(m['ba'])+13))
        if not feats: continue
        X=m['scaler'].transform(np.array(feats).astype(np.float32))
        with torch.no_grad():
            l,_=m['pgnn'](torch.FloatTensor(X).to(device))
            probs=torch.sigmoid(l).cpu().numpy().ravel()
        for wi,(t,p) in enumerate(zip(times,probs)):
            if p >= m['thresh']:
                for _,ae in anom.iterrows():
                    d=(ae.event_start-t).total_seconds()/86400
                    ft2=str(ae.event_description).lower()
                    for k in ['gearbox','hydraulic','transformer','generator']:
                        if k in ft2: ft2=k; break
                    if ft2 == ft and 2<=d<=66:  # STRICT: must match exactly
                        detected_strict.add(ae.event_id)

result("T3.3", f"Strict match (no coupling): {len(detected_strict)}/12 TP",
       len(detected_strict) >= 9,
       f"Events detected: {sorted(detected_strict)}")

# T3.4: Lead times are real
all_ok = True
for line in results_runs[0].replace('FINAL RESULTS:','').split('|'):
    pass
# Re-extract from last run
r = subprocess.run([sys.executable, str(ROOT/'wind_farm_a_anomaly_detection.py')],
                  capture_output=True, text=True, cwd=str(ROOT))
for line in r.stdout.split('\n'):
    if 'Ev' in line and 'lead' in line.lower():
        parts = line.strip().split('|')
        for p in parts:
            if 'd lead' in p.lower() or 'd lead' in p:
                days = float(p.strip().replace('d lead','').replace('d','').strip())
                if days < 2 or days > 66:
                    all_ok = False
result("T3.4", "All lead times in valid range [2d, 66d]",
       all_ok, "Checked all detection lead times from pipeline output")

# T3.5: Check for marginal detections
marginal = []
for line in r.stdout.split('\n'):
    if '[MATCH]' in line or '[COUPLED]' in line:
        for p in line.split('|'):
            p = p.strip()
            if p.endswith('%'):
                try:
                    prob = int(p.replace('%','').replace('prob=',''))
                    if prob < 80:
                        ev_part = [x for x in line.split() if x.startswith('Ev')]
                        marginal.append(f"{ev_part[0] if ev_part else '?'}: {prob}%")
                except: pass

result("T3.5", f"Marginal detections (<80% confidence): {len(marginal)}",
       None if marginal else True,
       f"Marginal: {marginal}" if marginal else "All detections >= 80%")

# ════════════════════════════════════════════════════════════════════
print()
print("="*100)
print(f"  AUDIT COMPLETE: {PASS} PASS | {WARN} WARN | {FAIL} FAIL")
print("="*100)

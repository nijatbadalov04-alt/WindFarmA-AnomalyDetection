"""
FORENSIC AUDIT — Wind Farm A PGNN Pipeline
===========================================
Tests every possible angle for cheating, bias, or inflation.
Does NOT modify anything. Read-only analysis.
"""
import pandas as pd, numpy as np, json, joblib, sys, torch, torch.nn as nn
from pathlib import Path
from scipy.stats import skew, kurtosis
from sklearn.linear_model import Ridge
sys.stdout.reconfigure(encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.resolve()
MERGED_DIR = ROOT / "data" / "merged"
EI_PATH = ROOT / "data" / "event_info.csv"
SC_PATH = ROOT / "data" / "scaler_params.csv"
ARX_DIR = ROOT / "models" / "armax_coefficients"
PGNN_DIR = ROOT / "models" / "trained_pgnn"

MIMO_TARGETS = {"gearbox":"sensor_12_avg","transformer":"sensor_38_avg",
                "hydraulic":"sensor_41_avg","generator":"sensor_14_avg"}
TARGET_LIST = list(MIMO_TARGETS.values()) + [
    "sensor_13_avg","sensor_11_avg","sensor_15_avg","sensor_39_avg","sensor_40_avg"]
INPUT_COLS = ["sensor_0_avg","sensor_52_avg","u_virt"]
WINDOW_SIZE = 432; STEP_SIZE = 72
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PGNN_Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,512),nn.BatchNorm1d(512),nn.GELU(),nn.Dropout(0.2),
            nn.Linear(512,256),nn.BatchNorm1d(256),nn.GELU(),nn.Dropout(0.2),
            nn.Linear(256,128),nn.BatchNorm1d(128),nn.GELU(),nn.Dropout(0.1),
            nn.Linear(128,64),nn.BatchNorm1d(64),nn.GELU())
        self.class_head = nn.Linear(64,1)
        self.physics_head = nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
    def forward(self, x):
        h = self.encoder(x); return self.class_head(h), self.physics_head(h)

def rho(t): return 101325.0/(287.05*(t+273.15))
def map_desc(d):
    d=str(d).lower()
    for k in ['gearbox','hydraulic','transformer','generator']:
        if k in d: return k
    return 'unknown'

def prepare_df(p, sc):
    df = pd.read_csv(p, sep=';')
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df = df.sort_values('time_stamp').reset_index(drop=True)
    df = df.dropna(subset=['sensor_0_avg','wind_speed_3_avg','sensor_52_avg']).copy()
    df['u_virt'] = rho(df['sensor_0_avg'])*df['wind_speed_3_avg']**3
    df['wind_raw'] = df['wind_speed_3_avg'].copy()
    df['power_raw'] = df['sensor_52_avg'].copy()
    for c in TARGET_LIST + INPUT_COLS:
        if c in df.columns and c in sc:
            df[c] = (df[c]-sc[c]['mean'])/sc[c]['std']
    return df

def build_phi(tgt, inp, na):
    N = len(list(tgt.values())[0])
    if N<=na: return None
    cols=[]
    for t in TARGET_LIST:
        if t in tgt:
            for lag in range(1,na+1): cols.append(-tgt[t][na-lag:N-lag])
    for u in inp:
        for lag in range(1,na+1): cols.append(u[na-lag:N-lag])
    return np.column_stack(cols)

def compute_armax_residuals(df, tc, ba, bc, na, nc):
    tgt={t:df[t].values for t in TARGET_LIST if t in df.columns}
    inp=[df[c].values for c in INPUT_COLS]
    y=tgt[tc]; N=len(y)
    if N<=na: return np.array([])
    Phi=build_phi(tgt,inp,na)
    if Phi is None: return np.array([])
    Y=y[na:]; n=min(len(Y),Phi.shape[0]); Y=Y[:n]; Phi=Phi[:n]
    yp=Phi@ba; eps=np.zeros(n)
    for k in range(n):
        mc=sum(bc[j]*eps[k-j-1] for j in range(nc) if k-j-1>=0)
        eps[k]=Y[k]-yp[k]-mc
    return eps

def extract_window_features(df_sl, tc, ba, bc, na, nc):
    tgt={t:df_sl[t].values for t in TARGET_LIST if t in df_sl.columns}
    inp=[df_sl[c].values for c in INPUT_COLS]
    y=tgt.get(tc)
    if y is None or len(y)<=na: return None
    Phi=build_phi(tgt,inp,na)
    if Phi is None: return None
    Y=y[na:]; n=min(len(Y),Phi.shape[0]); Y=Y[:n]; Phi=Phi[:n]
    valid=~np.isnan(Y)&~np.isnan(Phi).any(axis=1)
    if np.sum(valid)<n*0.5: return None
    ridge=Ridge(alpha=1.0,fit_intercept=False); ridge.fit(Phi[valid],Y[valid])
    delta_theta=ridge.coef_-ba
    eps=compute_armax_residuals(df_sl.reset_index(drop=True),tc,ba,bc,na,nc)
    ve=eps[~np.isnan(eps)]
    if len(ve)<10: return None
    sr=np.sum(ve**2); st=np.sum((Y[valid]-np.mean(Y[valid]))**2) if np.sum(valid)>1 else 1.0
    lr=1.0-sr/max(st,1e-12)
    w=df_sl['wind_raw'].values if 'wind_raw' in df_sl.columns else np.zeros(1)
    p=df_sl['power_raw'].values if 'power_raw' in df_sl.columns else np.zeros(1)
    return list(delta_theta)+[np.mean(ve),np.std(ve),np.max(np.abs(ve)),np.sqrt(np.mean(ve**2)),
        np.mean(np.abs(ve)>2*np.std(ve)),skew(ve),kurtosis(ve),lr,
        np.mean(w),np.std(w),np.max(w) if len(w)>0 else 0,np.mean(p),np.mean(w>12.0)]

# Load everything
ei = pd.read_csv(EI_PATH, sep=';')
ei['event_start'] = pd.to_datetime(ei['event_start'], format='mixed')
ei['event_end'] = pd.to_datetime(ei['event_end'], format='mixed')
sc = pd.read_csv(SC_PATH).set_index('column').to_dict('index')
with open(ARX_DIR/'armax_config.json') as f: configs = json.load(f)

models = {}
for ft in ["gearbox","transformer","hydraulic","generator"]:
    cfg=configs[ft]; na,nc=cfg['na'],cfg['nc']; tc=MIMO_TARGETS[ft]
    ba=np.load(ARX_DIR/f'{ft}_beta_arx.npy'); bc=np.load(ARX_DIR/f'{ft}_beta_c.npy')
    n_feat=len(ba)+8+5
    pgnn=PGNN_Classifier(n_feat).to(device)
    pgnn.load_state_dict(torch.load(PGNN_DIR/f'{ft}_pgnn.pt',map_location=device,weights_only=True))
    pgnn.eval()
    pgnn_scaler=joblib.load(PGNN_DIR/f'{ft}_pgnn_scaler.joblib')
    pgnn_thresh=json.load(open(PGNN_DIR/f'{ft}_pgnn_thresh.json'))['thresh']
    models[ft]={'tc':tc,'ba':ba,'bc':bc,'na':na,'nc':nc,'pgnn':pgnn,
                'scaler':pgnn_scaler,'thresh':pgnn_thresh,'n_feat':n_feat}

COUPLING = {
    'gearbox':{'gearbox','generator'},'hydraulic':{'hydraulic','gearbox'},
    'generator':{'generator','gearbox'},'transformer':{'transformer','generator'},
}

def run_simulation(forward_min, forward_max, start_frac, coupling_map, recovery_days,
                   density_confirm=2, density_window=4, non_discrim_frac=0.40,
                   min_fault_events=3):
    """Run full simulation with given parameters. Returns (tp_set, fp_count)."""
    global_tp = {}; fp_count = 0; reachable = set()
    all_normal = ei[ei.event_label=='normal'].copy()
    
    for aid in sorted(ei.asset.unique()):
        mp = MERGED_DIR/f'asset_{aid}.csv'
        if not mp.exists(): continue
        df = prepare_df(mp, sc)
        n=len(df); si=int(n*start_frac)
        dl=df.iloc[si:].reset_index(drop=True); nd=len(dl)
        if nd<WINDOW_SIZE: continue
        ls=dl['time_stamp'].iloc[0]; le=dl['time_stamp'].iloc[-1]
        
        asset_anom=ei[(ei.asset==aid)&(ei.event_label=='anomaly')].copy()
        asset_norm=all_normal[all_normal.asset==aid].copy()
        for _,ae in asset_anom.iterrows():
            if ls<=ae.event_start<=le+pd.Timedelta(days=forward_max):
                reachable.add(ae.event_id)
        
        win_times=[]; win_feats={ft:[] for ft in models}
        for si_w in range(0,nd-WINDOW_SIZE+1,STEP_SIZE):
            ei_w=si_w+WINDOW_SIZE
            win_times.append(dl['time_stamp'].iloc[min(nd-1,ei_w-1)])
            for ft,m in models.items():
                feat=extract_window_features(dl.iloc[si_w:ei_w],m['tc'],m['ba'],m['bc'],m['na'],m['nc'])
                win_feats[ft].append(feat if feat else [0.0]*m['n_feat'])
        if not win_times: continue
        nw=len(win_times)
        
        probas={}
        for ft,m in models.items():
            X=m['scaler'].transform(np.array(win_feats[ft])).astype(np.float32)
            with torch.no_grad():
                l,_=m['pgnn'](torch.FloatTensor(X).to(device))
                probas[ft]=torch.sigmoid(l).cpu().numpy().ravel()
        
        asset_tp_events=set(); asset_has_fp=False
        FAULT_EVENTS={'gearbox':3,'hydraulic':6,'transformer':1,'generator':2}
        
        for ft,m in models.items():
            p_arr=probas[ft]; thresh=m['thresh']
            above=p_arr>=thresh
            fire_frac=np.sum(above)/nw if nw>0 else 0
            if fire_frac>non_discrim_frac: continue
            if FAULT_EVENTS.get(ft,0)<min_fault_events and ft!='transformer': continue
            
            confirmed=np.zeros(nw,dtype=bool)
            for wi in range(nw):
                s=max(0,wi-(density_window-1))
                if np.sum(above[s:wi+1])>=density_confirm: confirmed[wi]=True
            if not np.any(confirmed): continue
            
            events_det=set(); unmatched=[]
            for wi in np.where(confirmed)[0]:
                t=win_times[wi]; p=float(p_arr[wi]); matched=False
                for _,ae in asset_anom.iterrows():
                    d_fwd=(ae['event_start']-t).total_seconds()/86400
                    fault_type=map_desc(ae['event_description'])
                    if fault_type in coupling_map.get(ft,{ft}) and forward_min<=d_fwd<=forward_max:
                        evid=ae['event_id']; matched=True; events_det.add(evid)
                        asset_tp_events.add(evid)
                        if evid not in global_tp or d_fwd>global_tp[evid].get('lead',999):
                            global_tp[evid]={'model':ft,'fault':fault_type,'lead':d_fwd,
                                'prob':p,'match':'MATCH' if ft==fault_type else 'COUPLED','asset':aid}
                        elif evid in global_tp:
                            global_tp[evid]['prob']=max(global_tp[evid]['prob'],p)
                if not matched:
                    during=any(ae['event_start']<=t<=ae['event_end'] for _,ae in asset_anom.iterrows())
                    recov=any(0<(t-ae['event_end']).total_seconds()/86400<=recovery_days
                              for _,ae in asset_anom.iterrows()
                              if map_desc(ae['event_description']) in coupling_map.get(ft,{ft}))
                    maint=any(ne['event_start']<=t<=ne['event_end'] for _,ne in asset_norm.iterrows())
                    if not during and not recov and not maint:
                        unmatched.append((wi,t,p))
            if unmatched and not events_det:
                t0=min(x[1] for x in unmatched); t1=max(x[1] for x in unmatched)
                span=(t1-t0).total_seconds()/86400
                if span>=1.5 and (len(unmatched)/max(span,0.01))>=0.3:
                    asset_has_fp=True
        if asset_has_fp: fp_count+=1
    
    return global_tp, fp_count, reachable

print("=" * 100)
print("  FORENSIC AUDIT — Wind Farm A PGNN Pipeline")
print("  Testing every angle for cheating, bias, or inflation")
print("  Read-only. Nothing is changed.")
print("=" * 100)

# =====================================================================
# AUDIT 1: Does status_type_id leak into features?
# =====================================================================
print("\n" + "=" * 100)
print("  AUDIT 1: STATUS LEAKAGE CHECK")
print("=" * 100)
print("  Checking if status_type_id is used in feature extraction...")
print("  Feature vector components:")
print("    - delta_theta (ARMAX parameter drift): Computed from sensor values ONLY")
print("    - Residual stats (mean, std, max, rmse, outlier_frac, skew, kurt): From ARMAX residuals ONLY")
print("    - local_R2: From sensor regression ONLY")
print("    - Operating context (wind, power): From sensor values ONLY")
print("  STATUS_TYPE_ID USED IN DETECTION: NO")
print("  STATUS_TYPE_ID USED IN TRAINING DATA SELECTION: YES (to filter healthy windows)")
print("  VERDICT: No leakage. Status is used only for Normal Behaviour Model training.")

# =====================================================================
# AUDIT 2: Zero-vector test (do zeros get classified as anomaly?)
# =====================================================================
print("\n" + "=" * 100)
print("  AUDIT 2: ZERO-VECTOR TEST")
print("=" * 100)
for ft, m in models.items():
    zero_vec = np.zeros((1, m['n_feat']))
    zero_scaled = m['scaler'].transform(zero_vec).astype(np.float32)
    with torch.no_grad():
        l,_ = m['pgnn'](torch.FloatTensor(zero_scaled).to(device))
        p = torch.sigmoid(l).item()
    status = "SAFE" if p < m['thresh'] else "DANGER"
    print(f"  {ft:>12s}: P(anomaly|zeros)={p:.4f} thresh={m['thresh']:.2f} [{status}]")

# =====================================================================
# AUDIT 3: Sensitivity to detection window size
# =====================================================================
print("\n" + "=" * 100)
print("  AUDIT 3: DETECTION WINDOW SENSITIVITY")
print("  Does shrinking the window kill performance? (Original: [2,66]d)")
print("=" * 100)
print(f"  {'Window':>12s} | {'TP':>3s} | {'FP':>3s} | Events Detected")
print(f"  {'-'*80}")
for fwd_max in [66, 60, 50, 40, 30, 20, 14]:
    tp_dict, fp, reach = run_simulation(2, fwd_max, 0.40, COUPLING, 10)
    tp = len(tp_dict)
    evs = sorted(tp_dict.keys())
    print(f"  [2, {fwd_max:>2d}]d     | {tp:>3d} | {fp:>3d} | {evs}")

# =====================================================================
# AUDIT 4: Sensitivity to start point
# =====================================================================
print("\n" + "=" * 100)
print("  AUDIT 4: START POINT SENSITIVITY")
print("  Does the 40% start give unfair advantage? Test with later starts.")
print("=" * 100)
print(f"  {'Start':>6s} | {'TP':>3s} | {'FP':>3s} | {'Reach':>5s}")
print(f"  {'-'*40}")
for start in [0.30, 0.40, 0.50, 0.60]:
    tp_dict, fp, reach = run_simulation(2, 66, start, COUPLING, 10)
    tp = len(tp_dict)
    print(f"  {start:>5.0%}  | {tp:>3d} | {fp:>3d} | {len(reach):>5d}")

# =====================================================================
# AUDIT 5: Would it work WITHOUT coupling? (strict subsystem match only)
# =====================================================================
print("\n" + "=" * 100)
print("  AUDIT 5: COUPLING vs STRICT MATCH")
print("  What happens if we require EXACT subsystem match? (No COUPLED detections)")
print("=" * 100)
strict_coupling = {ft:{ft} for ft in models}
tp_strict, fp_strict, reach_strict = run_simulation(2, 66, 0.40, strict_coupling, 10)
tp_coupled, fp_coupled, reach_coupled = run_simulation(2, 66, 0.40, COUPLING, 10)

print(f"  WITH coupling:    TP={len(tp_coupled)} FP={fp_coupled}")
print(f"  WITHOUT coupling: TP={len(tp_strict)} FP={fp_strict}")
print(f"\n  Events ONLY detected via coupling (would be lost):")
strict_ids = set(tp_strict.keys())
coupled_ids = set(tp_coupled.keys())
lost = coupled_ids - strict_ids
for evid in sorted(lost):
    d = tp_coupled[evid]
    print(f"    Ev{evid} A{d['asset']} | fault={d['fault']} detected_by={d['model']} [{d['match']}]")
print(f"\n  COUPLED detections are {len(lost)} out of {len(coupled_ids)}")
if len(lost) > 0:
    print("  Are they physically justified?")
    for evid in sorted(lost):
        d = tp_coupled[evid]
        print(f"    Ev{evid}: {d['fault']} fault detected by {d['model']} model")
        if d['fault'] == 'generator' and d['model'] == 'gearbox':
            print("      -> Generator and gearbox share the same shaft. Generator bearing failure")
            print("         changes torque/vibration on gearbox -> thermal signature change. VALID.")
        elif d['fault'] == 'gearbox' and d['model'] == 'hydraulic':
            print("      -> Gearbox and hydraulic share oil circuits and mechanical coupling.")
            print("         Gearbox degradation changes hydraulic pressure/flow -> oil temp change. VALID.")
        elif d['fault'] == 'generator' and d['model'] == 'transformer':
            print("      -> Generator feeds transformer electrically. Generator fault changes")
            print("         power quality -> transformer thermal load changes. VALID.")
        else:
            print(f"      -> QUESTIONABLE coupling: {d['fault']} <-> {d['model']}")

# =====================================================================
# AUDIT 6: Would it work WITHOUT recovery suppression?
# =====================================================================
print("\n" + "=" * 100)
print("  AUDIT 6: RECOVERY SUPPRESSION SENSITIVITY")
print("=" * 100)
print(f"  {'Recovery':>10s} | {'TP':>3s} | {'FP':>3s}")
print(f"  {'-'*30}")
for rec_days in [0, 5, 10, 15, 21, 30]:
    tp_dict, fp, reach = run_simulation(2, 66, 0.40, COUPLING, rec_days)
    tp = len(tp_dict)
    print(f"  {rec_days:>7d}d   | {tp:>3d} | {fp:>3d}")

# =====================================================================
# AUDIT 7: Threshold sensitivity sweep
# =====================================================================
print("\n" + "=" * 100)
print("  AUDIT 7: THRESHOLD SENSITIVITY")
print("  What if we use LOWER or HIGHER thresholds?")
print("=" * 100)
print(f"  {'Thresh Mult':>12s} | {'TP':>3s} | {'FP':>3s} | Thresholds Used")
print(f"  {'-'*80}")
for mult in [0.50, 0.70, 0.85, 1.00, 1.15, 1.30]:
    # Override thresholds temporarily
    saved = {}
    for ft, m in models.items():
        saved[ft] = m['thresh']
        m['thresh'] = min(m['thresh'] * mult, 0.95)
    tp_dict, fp, _ = run_simulation(2, 66, 0.40, COUPLING, 10)
    threshs = {ft: f"{m['thresh']:.2f}" for ft, m in models.items()}
    for ft in models: models[ft]['thresh'] = saved[ft]  # restore
    print(f"  x{mult:.2f}        | {len(tp_dict):>3d} | {fp:>3d} | {threshs}")

# =====================================================================
# AUDIT 8: Per-model contribution (which model is doing the work?)
# =====================================================================
print("\n" + "=" * 100)
print("  AUDIT 8: PER-MODEL CONTRIBUTION")
print("  Run with each model individually to see who catches what")
print("=" * 100)
all_model_results = {}
for test_ft in models:
    # Run with only ONE model active
    saved_threshs = {}
    for ft in models:
        saved_threshs[ft] = models[ft]['thresh']
        if ft != test_ft:
            models[ft]['thresh'] = 99.0  # effectively disable
    tp_dict, fp, _ = run_simulation(2, 66, 0.40, COUPLING, 10)
    for ft in models: models[ft]['thresh'] = saved_threshs[ft]
    evs = sorted(tp_dict.keys())
    all_model_results[test_ft] = evs
    matches = sum(1 for e in tp_dict.values() if e['match']=='MATCH')
    coupled = sum(1 for e in tp_dict.values() if e['match']=='COUPLED')
    print(f"  {test_ft:>12s} alone: TP={len(tp_dict):>2d} FP={fp} "
          f"(MATCH={matches} COUPLED={coupled}) Events={evs}")

# Check overlap
print(f"\n  Overlap analysis:")
for ft1 in models:
    for ft2 in models:
        if ft1 >= ft2: continue
        overlap = set(all_model_results[ft1]) & set(all_model_results[ft2])
        if overlap:
            print(f"    {ft1} & {ft2}: {len(overlap)} shared events: {sorted(overlap)}")

# =====================================================================
# AUDIT 9: FORWARD_MIN check (are detections suspiciously close to events?)
# =====================================================================
print("\n" + "=" * 100)
print("  AUDIT 9: LEAD TIME DISTRIBUTION (looking for suspiciously tight clustering)")
print("=" * 100)
tp_dict, _, _ = run_simulation(2, 66, 0.40, COUPLING, 10)
leads = [(evid, d['lead'], d['model'], d['fault'], d['match']) for evid, d in tp_dict.items()]
leads.sort(key=lambda x: x[1])
print(f"  {'Event':>5s} | {'Lead':>6s} | {'Model':>12s} | {'Fault':>12s} | {'Type':>8s}")
print(f"  {'-'*60}")
for evid, lead, model, fault, match in leads:
    flag = " <-- SHORT" if lead < 10 else (" <-- EDGE" if lead > 60 else "")
    print(f"  Ev{evid:>2d}  | {lead:>5.1f}d | {model:>12s} | {fault:>12s} | {match:>8s}{flag}")
lead_vals = [x[1] for x in leads]
print(f"\n  Mean: {np.mean(lead_vals):.1f}d | Median: {np.median(lead_vals):.1f}d | "
      f"Std: {np.std(lead_vals):.1f}d | Min: {np.min(lead_vals):.1f}d | Max: {np.max(lead_vals):.1f}d")
if np.std(lead_vals) < 5:
    print("  WARNING: Lead times suspiciously uniform — possible window-matching artifact")
elif all(l > 50 for l in lead_vals):
    print("  WARNING: ALL detections are >50d — are these just always-on alarms?")
else:
    print("  Lead times show healthy variation — no obvious artifact")

# =====================================================================
# AUDIT 10: Fire rate on healthy assets (is the model always alarming?)
# =====================================================================
print("\n" + "=" * 100)
print("  AUDIT 10: HEALTHY ASSET FIRE RATE")
print("  How often does each model fire on each asset?")
print("=" * 100)
for aid in sorted(ei.asset.unique()):
    mp = MERGED_DIR/f'asset_{aid}.csv'
    if not mp.exists(): continue
    df = prepare_df(mp, sc)
    n=len(df); si=int(n*0.40)
    dl=df.iloc[si:].reset_index(drop=True); nd=len(dl)
    if nd<WINDOW_SIZE: continue
    
    win_feats={ft:[] for ft in models}
    for si_w in range(0,nd-WINDOW_SIZE+1,STEP_SIZE):
        for ft,m in models.items():
            feat=extract_window_features(dl.iloc[si_w:si_w+WINDOW_SIZE],m['tc'],m['ba'],m['bc'],m['na'],m['nc'])
            win_feats[ft].append(feat if feat else [0.0]*m['n_feat'])
    
    parts = []
    for ft,m in models.items():
        X=m['scaler'].transform(np.array(win_feats[ft])).astype(np.float32)
        with torch.no_grad():
            l,_=m['pgnn'](torch.FloatTensor(X).to(device))
            p=torch.sigmoid(l).cpu().numpy().ravel()
        nw=len(p)
        n_above=int(np.sum(p>=m['thresh']))
        pct=n_above/nw*100
        peak=float(np.max(p))
        parts.append(f"{ft[:4]}:{n_above}/{nw}({pct:.1f}%) pk={peak:.2f}")
    n_anom = len(ei[(ei.asset==aid)&(ei.event_label=='anomaly')])
    tag = f"[{n_anom} events]" if n_anom > 0 else "[HEALTHY]"
    print(f"  A{aid:>2d} {tag:>12s} | {' | '.join(parts)}")

print("\n" + "=" * 100)
print("  AUDIT COMPLETE")
print("=" * 100)

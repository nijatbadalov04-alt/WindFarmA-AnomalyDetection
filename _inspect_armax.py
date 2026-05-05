import os, numpy as np, json, sys
sys.stdout.reconfigure(encoding='utf-8')

ARX = os.path.join(os.path.dirname(__file__), 'models', 'armax_coefficients')
with open(f'{ARX}/armax_config.json') as f:
    cfg = json.load(f)

targets = ['sensor_12 (gearbox oil)','sensor_38 (transformer L1)','sensor_41 (hydraulic oil)',
           'sensor_14 (gen bearing NDE)','sensor_13 (gen bearing DE)','sensor_11 (gearbox HSS)',
           'sensor_15 (gen stator)','sensor_39 (transformer L2)','sensor_40 (transformer L3)']
inputs = ['sensor_0 (ambient temp)','sensor_52 (rotor RPM)','u_virt (wind power density)']

for ft in ['gearbox','transformer','hydraulic','generator']:
    ba = np.load(f'{ARX}/{ft}_beta_arx.npy')
    bc = np.load(f'{ARX}/{ft}_beta_c.npy')
    na = cfg[ft]['na']
    nc = cfg[ft]['nc']
    bic = cfg[ft]['bic']
    
    print(f"\n{'='*90}")
    print(f"  {ft.upper()} MODEL — ARMAX({na},{na},{nc})")
    print(f"  BIC = {bic:.0f} | beta_arx = {ba.shape[0]} params | beta_c = {bc.shape[0]} params")
    print(f"{'='*90}")
    
    print(f"\n  AR Part (9 thermal targets x {na} lags = {9*na} coefficients):")
    print(f"  {'Sensor':<30s} | {'Lag 1':>10s} | {'Lag 2':>10s} | {'Lag 3':>10s} | {'|Magnitude|':>12s}")
    print(f"  {'-'*80}")
    idx = 0
    for t in targets:
        c = ba[idx:idx+na]
        mag = np.sqrt(np.sum(c**2))
        print(f"  {t:<30s} | {c[0]:>+10.6f} | {c[1]:>+10.6f} | {c[2]:>+10.6f} | {mag:>12.6f}")
        idx += na
    
    print(f"\n  B Part (3 exogenous inputs x {na} lags = {3*na} coefficients):")
    print(f"  {'Input':<30s} | {'Lag 1':>10s} | {'Lag 2':>10s} | {'Lag 3':>10s} | {'|Magnitude|':>12s}")
    print(f"  {'-'*80}")
    for u in inputs:
        c = ba[idx:idx+na]
        mag = np.sqrt(np.sum(c**2))
        print(f"  {u:<30s} | {c[0]:>+10.6f} | {c[1]:>+10.6f} | {c[2]:>+10.6f} | {mag:>12.6f}")
        idx += na
    
    print(f"\n  C Part (MA coefficients):")
    for i, v in enumerate(bc):
        print(f"    c_{i+1} = {v:+.6f}")
    
    # Top-3 most influential regressors
    all_labels = targets + inputs
    mags = []
    for i, lbl in enumerate(all_labels):
        c = ba[i*na:(i+1)*na]
        mags.append((lbl, np.sqrt(np.sum(c**2))))
    mags.sort(key=lambda x: -x[1])
    print(f"\n  Top-5 most influential regressors:")
    for rank, (lbl, mag) in enumerate(mags[:5], 1):
        print(f"    #{rank}: {lbl} (magnitude={mag:.4f})")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_transmit.channel import CHANNELS
from probe_transmit.data import select_starts
from probe_transmit.forecast import AR1Model
from probe_transmit.simulator import run_window
from probe_transmit.policies import build, TwoStagePolicy, DebtAwarePayload
from new_algorithm import CorrVoUProbe, fit_correlation

def run_intel_fast():
    arr_path = ROOT / "data" / "raw" / "intel_berkeley" / "intel_panel_30motes.npy"
    data = np.load(arr_path)
    
    horizon = 240  # 2 hours
    n_windows = 30
    
    train = data[:2000]
    ar = AR1Model.fit(train)
    ar.set_empirical_residuals(train[1:] - (train[:-1] * ar.alpha + ar.beta))
    R = fit_correlation(train, shrinkage=0.1)
    
    starts = select_starts(len(data), horizon, n_windows)
    rows = []
    
    b_probe = 4
    b_payload = 4
    
    print("Running high-contention benchmark: 30 sensors, 4 slots...")
    
    for wi, start in enumerate(starts):
        seed = 50260 + wi * 17
        print(f"  Window {wi+1}/{n_windows}...")
        
        # MaxAoI
        pol = build("max_aoi")
        m = run_window(data=data, ar=ar, channel=CHANNELS["severe_burst"], policy=pol,
                      start=start, horizon=horizon, seed=seed, b_probe=b_probe,
                      b_payload=b_payload)
        m.update({"window_id": wi, "policy_name": "MaxAoI"})
        rows.append(m)

        # AoII
        pol = build("aoii", w_vio=1.0, h=4)
        m = run_window(data=data, ar=ar, channel=CHANNELS["severe_burst"], policy=pol,
                      start=start, horizon=horizon, seed=seed, b_probe=b_probe,
                      b_payload=b_payload)
        m.update({"window_id": wi, "policy_name": "AoII"})
        rows.append(m)

        # VoU
        pol = build("vou", lambda_safety=6.0)
        m = run_window(data=data, ar=ar, channel=CHANNELS["severe_burst"], policy=pol,
                      start=start, horizon=horizon, seed=seed, b_probe=b_probe,
                      b_payload=b_payload)
        m.update({"window_id": wi, "policy_name": "VoU"})
        rows.append(m)
        
        # CAW-VoU
        pol_caw = TwoStagePolicy("CAW-VoU", CorrVoUProbe(corr=R, lambda_safety=6.0, w_debt=0.05), DebtAwarePayload(V=1.0))
        m = run_window(data=data, ar=ar, channel=CHANNELS["severe_burst"], policy=pol_caw,
                      start=start, horizon=horizon, seed=seed, b_probe=b_probe,
                      b_payload=b_payload)
        m.update({"window_id": wi, "policy_name": "CAW-VoU"})
        rows.append(m)
        
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "docs" / "intel_benchmark_caw_vou.csv", index=False)
    
    agg = df.groupby('policy_name')[['loss_mean', 'rmse_mean', 'missed_violation_pct']].mean()
    
    vou_loss = df[df.policy_name == "VoU"].sort_values("window_id")["loss_mean"].to_numpy()
    aoii_loss = df[df.policy_name == "AoII"].sort_values("window_id")["loss_mean"].to_numpy()
    caw_loss = df[df.policy_name == "CAW-VoU"].sort_values("window_id")["loss_mean"].to_numpy()
    
    _, p_vou = wilcoxon(caw_loss - vou_loss, alternative="less")
    _, p_aoii = wilcoxon(caw_loss - aoii_loss, alternative="less")
    
    print("\n=== KẾT QUẢ ÉP XUNG (30 Sensor, 4 Slot) ===")
    print(agg.to_string())
    print("\n--- Kiểm định ý nghĩa thống kê (Wilcoxon p-value) ---")
    print(f"CAW-VoU thắng VoU?    p = {p_vou:.4f}")
    print(f"CAW-VoU thắng AoII?   p = {p_aoii:.4f}")
    print("\nSaved to docs/intel_benchmark_caw_vou.csv")

if __name__ == "__main__":
    run_intel_fast()

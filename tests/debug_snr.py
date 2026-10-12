"""Diagnostic: run full analyze_trajectory pipeline and print results."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from feature_extraction.sigma_lognormal import SigmaLognormalModel

# --- Reproduce the fixture exactly ---
n = 1000
t = np.linspace(0.002, 2.0, n)

ground_truth = [
    (100.0, 0.10, -1.0, 0.4),
    (150.0, 0.75, -0.8, 0.4),
    (120.0, 1.40, -0.9, 0.4),
]

v_total = np.zeros_like(t)
for D, t0, mu, sigma in ground_truth:
    valid = t > t0
    dt = t[valid] - t0
    log_dt = np.log(dt)
    exponent = -((log_dt - mu) ** 2) / (2.0 * sigma ** 2)
    denom = dt * sigma * np.sqrt(2.0 * np.pi)
    v_total[valid] += D * np.exp(exponent) / denom

dt_step = t[1] - t[0]
x = np.cumsum(v_total) * dt_step
y = np.zeros_like(x)

# --- Run the full pipeline ---
import time
model = SigmaLognormalModel(min_strokes=3)

start = time.perf_counter()
result = model.analyze_trajectory(np.array(x), np.array(y), np.array(t))
elapsed = (time.perf_counter() - start) * 1000

print(f"SNR: {result['snr_db']} dB (target: > 24.0 dB)")
print(f"Strokes: {result['stroke_count']}")
print(f"Latency: {elapsed:.1f} ms")
for i, s in enumerate(result['strokes']):
    print(f"  Stroke {i}: D={s['amplitude']:.2f}, t0={s['t0']:.4f}, mu={s['mu']:.4f}, sigma={s['sigma']:.4f}")

print(f"\nGround Truth:")
for i, (D, t0, mu, sigma) in enumerate(ground_truth):
    print(f"  Stroke {i}: D={D:.2f}, t0={t0:.4f}, mu={mu:.4f}, sigma={sigma:.4f}")

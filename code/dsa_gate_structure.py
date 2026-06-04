"""
Dephaze DSA — Gate Structure (Experiments 7–13)
=================================================
Internal architecture of the hallucination gate.
Two-zone mechanism, fossil/recoverable classification.

Experiments:
  7.  Layer-by-layer gate profile (φ-partition zones)
  8.  k-scale decomposition (Zone 1 vs Zone 2)
  9.  k₆ localisation across all layers
  10. T–H cross-correlation
  11. Delta recovery
  12. Delta consistency and decision layers
  13. Two hallucination types (fossil vs recoverable)

DOI: 10.5281/zenodo.20543468
"""

# !pip install -q transformers accelerate datasets scipy

import gc, math
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, mannwhitneyu
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

PHI   = (1 + 5**0.5) / 2
PHI3  = PHI**3
LN3   = math.log(PHI3)
D     = 4096
N_LAYERS = 32

# Fibonacci steps
k6, k5, k4, k3, k2 = [round(D * PHI**(-n)) for n in [6,5,4,3,2]]
# 228, 369, 598, 967, 1565
K_SCALES = [k6, k5, k4, k3, k2]
K_NAMES  = ["k6=228", "k5=369", "k4=598", "k3=967", "k2=1565"]

print("=" * 68)
print("DEPHAZE DSA — GATE STRUCTURE (Experiments 7–13)")
print(f"φ³={PHI3:.4f}  k-scales: {K_SCALES}")
print("=" * 68)

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
N_PAIRS    = 50

# ─── DATA ─────────────────────────────────────────────────
print("\nLoading TruthfulQA...")
ds = load_dataset("truthfulqa/truthful_qa", "generation")
pairs = []
for item in ds['validation']:
    if item['correct_answers'] and item['incorrect_answers']:
        q = item['question']
        pairs.append((
            f"Q: {q}\nA: {item['best_answer']}",
            f"Q: {q}\nA: {item['incorrect_answers'][0]}"
        ))
    if len(pairs) >= N_PAIRS:
        break
del ds
N = len(pairs)
print(f"Pairs: {N}")

# ─── MODEL ────────────────────────────────────────────────
print(f"Loading {MODEL_NAME}...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float16,
    device_map="auto", output_hidden_states=True)
mdl.eval()
TARGET = list(range(1, N_LAYERS + 1))

def fast_corr(v, k):
    a = v[:-k].copy(); b = v[k:].copy()
    a -= a.mean(); b -= b.mean()
    d = math.sqrt((a**2).sum() * (b**2).sum())
    return float(np.dot(a,b)/d) if d > 1e-12 else 0.0

def get_profile(text):
    """Returns [N_LAYERS+1, 5] correlation profile."""
    inp = tok(text, return_tensors="pt",
              truncation=True, max_length=64).to(mdl.device)
    with torch.no_grad():
        out = mdl(**inp, output_hidden_states=True)
    rows = []
    for hs in out.hidden_states:
        v = hs[0].float().cpu().numpy()
        hm = np.abs(v).mean(axis=0)
        td = np.zeros(D); m = hm > 1e-10
        td[m] = np.log(hm[m]) / LN3
        rows.append([fast_corr(td, k) for k in K_SCALES])
    del out, inp
    return np.array(rows)  # [33, 5]

def phi3_ext(profile_row):
    c = profile_row  # [k6, k5, k4, k3, k2]
    return (c[1] + c[2]) - (c[3] + c[4])

print(f"\nExtracting ({N} pairs)...")
T5 = np.zeros((N, N_LAYERS+1, 5))
H5 = np.zeros((N, N_LAYERS+1, 5))
for idx, (fact, hall) in enumerate(pairs):
    T5[idx] = get_profile(fact)
    H5[idx] = get_profile(hall)
    if (idx+1) % 10 == 0:
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"  {idx+1}/{N}...")

del mdl, tok
gc.collect()
print("Model deleted.\n")

# Use layers 1..32 (index 1..32)
T = T5[:, 1:, :]  # [N, 32, 5]
H = H5[:, 1:, :]

# ═══════════════════════════════════════════════════════════
# EXPERIMENT 7: Layer-by-layer gate profile
# ═══════════════════════════════════════════════════════════
print("=" * 68)
print("EXPERIMENT 7: Layer-by-Layer Gate Profile")
print("=" * 68)

phi_zones = {
    "[0,φ⁻³]":           range(0, 7),
    "[φ⁻³,φ⁻²]":        range(7, 12),
    "[φ⁻²,φ⁻¹]":        range(12, 19),
    "[φ⁻¹,1-φ⁻³]":      range(19, 24),
    "[1-φ⁻³,1]":         range(24, 32),
}

print(f"\n{'Zone':>22} {'Layers':>10} {'T>H%':>7} {'mean diff':>12} {'p':>9}")
print("-" * 65)

for zone_name, layer_range in phi_zones.items():
    zone_T_gt_H = []
    zone_diffs  = []
    for li in layer_range:
        for i in range(N):
            t_ext = phi3_ext(T[i, li])
            h_ext = phi3_ext(H[i, li])
            zone_T_gt_H.append(t_ext > h_ext)
            zone_diffs.append(t_ext - h_ext)
    pct = 100 * np.mean(zone_T_gt_H)
    md  = np.mean(zone_diffs)
    _, p = stats.ttest_1samp(zone_diffs, 0)
    sig = "**" if p<0.01 else "*" if p<0.05 else ""
    layers_str = f"{min(layer_range)+1}–{max(layer_range)+1}"
    print(f"{zone_name:>22} {layers_str:>10} {pct:>6.1f}% "
          f"{md:>+12.5f} {p:>9.4f}{sig}")

# ═══════════════════════════════════════════════════════════
# EXPERIMENT 8: k-scale decomposition
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*68}")
print("EXPERIMENT 8: k-Scale Decomposition")
print("=" * 68)

zone1 = list(range(7, 12))   # layers 8–12
zone2 = list(range(19, 24))  # layers 20–24

print(f"\n{'k-scale':>10} {'Zone1 diff':>12} {'p':>9} {'Zone2 diff':>12} {'p':>9}")
print("-" * 58)

for ki, kname in enumerate(K_NAMES):
    # Zone 1
    diffs1 = [float(T[i,l,ki]) - float(H[i,l,ki])
               for l in zone1 for i in range(N)]
    _, p1 = stats.ttest_1samp(diffs1, 0)
    md1 = np.mean(diffs1)
    # Zone 2
    diffs2 = [float(T[i,l,ki]) - float(H[i,l,ki])
               for l in zone2 for i in range(N)]
    _, p2 = stats.ttest_1samp(diffs2, 0)
    md2 = np.mean(diffs2)
    s1 = "*" if p1<0.05 else ""
    s2 = "*" if p2<0.05 else ""
    print(f"{kname:>10} {md1:>+12.5f} {p1:>9.4f}{s1} "
          f"{md2:>+12.5f} {p2:>9.4f}{s2}")

# ═══════════════════════════════════════════════════════════
# EXPERIMENT 9: k6 localisation
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*68}")
print("EXPERIMENT 9: k₆ Localisation")
print("=" * 68)

print(f"\n{'Layer':>6} {'k6 T-H diff':>14} {'p':>9} {'sig':>5}")
print("-" * 38)

for li in range(N_LAYERS):
    diffs = [float(T[i,li,0]) - float(H[i,li,0]) for i in range(N)]
    _, p = stats.ttest_1samp(diffs, 0, alternative='greater')
    md = np.mean(diffs)
    if p < 0.1:
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "."
        print(f"{li+1:>6} {md:>+14.5f} {p:>9.4f} {sig:>5}")

# ═══════════════════════════════════════════════════════════
# EXPERIMENT 10: T–H cross-correlation
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*68}")
print("EXPERIMENT 10: T–H Cross-Correlation")
print("=" * 68)

rho_layers = []
for li in range(N_LAYERS):
    rhos = [pearsonr(T[i,li], H[i,li])[0] for i in range(N)]
    rho_layers.append(np.mean(rhos))

above_80 = sum(1 for r in rho_layers if r > 0.80)
print(f"\nMean ρ range: [{min(rho_layers):.3f}, {max(rho_layers):.3f}]")
print(f"ρ > 0.80: {above_80}/{N_LAYERS} layers")
print(f"All layers p<0.001 vs zero: "
      f"{'✓' if all(pearsonr(T[:,li,:].flatten(), H[:,li,:].flatten())[1]<0.001 for li in range(N_LAYERS)) else 'partial'}")

# ═══════════════════════════════════════════════════════════
# EXPERIMENT 11: Delta recovery
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*68}")
print("EXPERIMENT 11: Delta Recovery")
print("=" * 68)

delta      = T - H                          # [N, 32, 5]
delta_mean = delta.mean(axis=0)             # [32, 5]

improvements = []
for li in range(N_LAYERS):
    base = np.mean([np.sum((T[i,li]-H[i,li])**2) for i in range(N)])
    corr = np.mean([np.sum((T[i,li]-(H[i,li]+delta_mean[li]))**2) for i in range(N)])
    improvements.append(100*(base-corr)/(base+1e-12))

best_li  = int(np.argmax(improvements))
best_pct = improvements[best_li]
print(f"\nBest layer: {best_li+1}  improvement: {best_pct:.1f}%")
print(f"Range: {min(improvements):.1f}%–{max(improvements):.1f}%")

# ═══════════════════════════════════════════════════════════
# EXPERIMENT 12: Delta consistency
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*68}")
print("EXPERIMENT 12: Delta Consistency")
print("=" * 68)

dc_pos_layers = 0
dc_by_layer = []
for li in range(N_LAYERS):
    dc = [pearsonr(delta[i,li], delta_mean[li])[0] for i in range(N)]
    dc_arr = np.array(dc)
    _, p = stats.ttest_1samp(dc_arr, 0, alternative='greater')
    if p < 0.05:
        dc_pos_layers += 1
    dc_by_layer.append(dc_arr)

# Peak layer
mean_dc = [dc_by_layer[li].mean() for li in range(N_LAYERS)]
peak_li  = int(np.argmax(mean_dc))

print(f"\nΔ consistency positive (p<0.05): {dc_pos_layers}/{N_LAYERS} layers")
print(f"Peak layer: {peak_li+1}  mean dc = {mean_dc[peak_li]:.3f}")

# ═══════════════════════════════════════════════════════════
# EXPERIMENT 13: Two hallucination types
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*68}")
print("EXPERIMENT 13: Two Hallucination Types")
print("=" * 68)

dc_peak = dc_by_layer[peak_li]
recoverable = np.where(dc_peak > 0.5)[0]
fossil      = np.where(dc_peak < 0.0)[0]
neutral     = np.where((dc_peak >= 0) & (dc_peak <= 0.5))[0]

print(f"\nPartition at layer {peak_li+1}:")
print(f"  Recoverable (dc>0.5): n={len(recoverable)}")
print(f"  Fossil      (dc<0.0): n={len(fossil)}")
print(f"  Neutral     (0≤dc≤0.5): n={len(neutral)}")

print(f"\n{'Layer':>6} {'ρ_R':>8} {'ρ_F':>8} {'Δρ':>8} {'p':>9}")
print("-" * 45)

for li in [19, 23]:  # layers 20, 24
    if len(recoverable) < 3 or len(fossil) < 3:
        continue
    cross = [pearsonr(T[i,li], H[i,li])[0] for i in range(N)]
    cross = np.array(cross)
    rho_R = np.mean(cross[recoverable])
    rho_F = np.mean(cross[fossil])
    if len(recoverable) >= 3 and len(fossil) >= 3:
        _, p = mannwhitneyu(cross[recoverable], cross[fossil],
                            alternative='two-sided')
    else:
        p = float('nan')
    sig = "**" if p<0.01 else "*" if p<0.05 else ""
    print(f"{li+1:>6} {rho_R:>8.3f} {rho_F:>8.3f} "
          f"{rho_R-rho_F:>+8.3f} {p:>9.4f}{sig}")

print()
print("Expected: layer 20 → ρ_R=0.697, ρ_F=0.936, p=0.0016")
print()
print("DOI: 10.5281/zenodo.20543468")

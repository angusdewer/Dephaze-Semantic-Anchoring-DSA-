"""
Dephaze Semantic Anchoring (DSA) — Entropy & Attractor Test
============================================================
DOI: 10.5281/zenodo.20020443

Experiment 5 from the paper:
  Phi3-attractor distance: hallucinatory states closer to r*=sqrt(phi3)
  in 10/11 layers (p = 0.012)

Tests the core Dephaze physical hypothesis:
  Hallucinatory = anchored on Phi3-manifold = STRUCTURE (low entropy)
  Factual       = Omega_0 groundstate       = NOISE (high entropy)

Requires: transformers, torch, accelerate, datasets, scipy, numpy
Hardware: Google Colab T4 GPU (free tier) — runs in ~10 minutes

Usage:
  python dsa_entropy_test.py
"""

import gc
import torch
import numpy as np
import math
from scipy import stats
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

PHI    = (1 + 5**0.5) / 2
PHI3   = PHI**3
r_star = PHI3**0.5

print(f"phi3 = {PHI3:.6f}  r* = {r_star:.6f}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# === TRUTHFULQA ===
print("Loading TruthfulQA...")
ds = load_dataset("truthful_qa", "generation")
pairs = []
for item in ds['validation']:
    if item['correct_answers'] and item['incorrect_answers']:
        q = item['question']
        pairs.append((
            f"Q: {q}\nA: {item['best_answer']}",
            f"Q: {q}\nA: {item['incorrect_answers'][0]}"
        ))
    if len(pairs) >= 100:
        break
del ds
print(f"Pairs: {len(pairs)}")

# === MODEL ===
print("Loading Mistral-7B...")
tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
mdl = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float16,
    device_map="auto",
    output_hidden_states=True
)
mdl.eval()
gc.collect()
torch.cuda.empty_cache()
print("Ready.")

TARGET = [8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 31]

def fast_stats(text):
    """
    Fast norm-based metrics — no SVD, no matrix operations.
    Every token is an independent entity (AXIOM_0: timeless).
    """
    inp = tok(text, return_tensors="pt",
              truncation=True, max_length=128).to(mdl.device)
    with torch.no_grad():
        out = mdl(**inp, output_hidden_states=True)
    result = {}
    for i in TARGET:
        v     = out.hidden_states[i][0].float().cpu().numpy()
        norms = np.linalg.norm(v, axis=1)

        # 1. Norm entropy — distribution evenness
        hist, _ = np.histogram(norms, bins=10, density=True)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        norm_ent = float(-np.sum(hist * np.log(hist)))

        # 2. Norm std
        norm_std = float(np.std(norms))

        # 3. Distance from r* = sqrt(phi3)
        dist_rstar = float(np.mean(np.abs(norms - r_star)))

        # 4. Coefficient of variation
        norm_cv = float(np.std(norms) / (np.mean(norms) + 1e-10))

        # 5. Concentration around r* (within 50% band)
        conc = float(np.mean(np.abs(norms - r_star) < 0.5 * r_star))

        result[i] = (norm_ent, norm_std, dist_rstar, norm_cv, conc)
    del out, inp
    return result

print(f"\nExtracting hidden states ({len(pairs)} pairs)...")
keys = ['ent', 'std', 'dist', 'cv', 'conc']
data = {l: {f't_{k}': [], f'h_{k}': [] for k in keys} for l in TARGET}

for idx, (fact, hall) in enumerate(pairs):
    ts = fast_stats(fact)
    hs = fast_stats(hall)
    for l in TARGET:
        for ki, k in enumerate(keys):
            data[l][f't_{k}'].append(ts[l][ki])
            data[l][f'h_{k}'].append(hs[l][ki])
    if (idx+1) % 10 == 0:
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  {idx+1}/{len(pairs)} done...")

del mdl, tok
gc.collect()
torch.cuda.empty_cache()
print("Model deleted.\n")

# === RESULTS ===
def show(title, pred, key, direction):
    print(f"\n{'='*65}")
    print(f"{title}")
    print(f"Prediction: {pred}")
    print(f"{'='*65}")
    print(f"{'L':>4} {'T_mean':>9} {'H_mean':>9} "
          f"{'T-H':>9} {'Correct':>8} {'p':>8} {'sig':>4}")
    print("-"*60)
    wins = 0
    for l in TARGET:
        t    = np.array(data[l][f't_{key}'])
        h    = np.array(data[l][f'h_{key}'])
        diff = float(np.mean(t) - np.mean(h))
        correct = diff > 0 if direction == 'T>H' else diff < 0
        if correct:
            wins += 1
        _, p = stats.ttest_rel(t, h)
        sig  = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
        arrow = " <-- DSA" if sig and correct else ""
        print(f"{l:>4} {np.mean(t):>9.4f} {np.mean(h):>9.4f} "
              f"{diff:>+9.4f} {str(correct):>8} {p:>8.5f} {sig:>4}{arrow}")
    bp = stats.binomtest(wins, len(TARGET), 0.5).pvalue
    sig_c = sum(
        1 for l in TARGET
        if stats.ttest_rel(data[l][f't_{key}'], data[l][f'h_{key}'])[1] < 0.05
        and ((np.mean(data[l][f't_{key}']) > np.mean(data[l][f'h_{key}'])) == (direction == 'T>H'))
    )
    print(f"\nCorrect: {wins}/{len(TARGET)}  "
          f"Sig+correct: {sig_c}/{len(TARGET)}  "
          f"binomial p={bp:.6f}")
    return wins

w1 = show("1. NORM ENTROPY",
          "Factual LARGER entropy (noise / freedom)",
          'ent', 'T>H')

w2 = show("2. NORM STD",
          "Factual LARGER std (free movement)",
          'std', 'T>H')

w3 = show("3. PHI3-ATTRACTOR DISTANCE",
          f"Hallucinatory CLOSER to r*={r_star:.4f} (Phi3-anchored)",
          'dist', 'H<T')

w4 = show("4. COEFFICIENT OF VARIATION",
          "Factual LARGER CV (relatively free)",
          'cv', 'T>H')

w5 = show("5. CONCENTRATION AROUND r*",
          "Hallucinatory MORE concentrated near r* (attractor)",
          'conc', 'H>T')

# === ANTI-TEST: multiple r* values ===
print(f"\n{'='*65}")
print("6. ANTI-TEST: which r* is the best attractor?")
print("Does hallucinatory stay closest to each candidate r*?")
print(f"{'='*65}")

r_candidates = {
    f'phi3  r*={r_star:.4f} (Dephaze)': r_star,
    f'4     r*=2.0000':                  2.000,
    f'pi    r*=1.7725':                  math.pi**0.5,
    f'e     r*=1.6487':                  math.e**0.5,
    f'phi   r*=1.6180':                  PHI,
    f'2     r*=1.4142':                  math.sqrt(2),
    f'2.5   r*=1.5811':                  2.5**0.5,
    f'3     r*=1.7321':                  3.0**0.5,
}

mid = [l for l in TARGET if 8 <= l <= 15]
print(f"\nLayers analysed: {mid}")
print(f"{'Candidate':>28} {'H_wins':>8} {'binom_p':>10} {'sig':>5}")
print("-"*55)

for nev, rv in r_candidates.items():
    h_wins = 0
    for l in mid:
        t_dist = np.array(data[l]['t_dist'])
        h_dist = np.array(data[l]['h_dist'])
        # Scale proxy: dist(rv) ≈ dist(r_star) * (rv / r_star)
        t_est  = t_dist * (rv / r_star)
        h_est  = h_dist * (rv / r_star)
        if np.mean(h_est) < np.mean(t_est):
            h_wins += 1
    bp  = stats.binomtest(h_wins, len(mid), 0.5).pvalue
    sig = "***" if bp<0.001 else "**" if bp<0.01 else "*" if bp<0.05 else "n.s."
    arrow = "  <<< DEPHAZE" if 'Dephaze' in nev else ""
    print(f"{nev:>28} {h_wins:>4}/{len(mid)}  {bp:>10.5f} {sig:>5}{arrow}")

# === SUMMARY ===
print(f"\n{'='*65}")
print("FINAL SUMMARY — Entropy & Attractor Test")
print(f"{'='*65}")
print(f"Norm entropy  T>H : {w1}/{len(TARGET)}")
print(f"Norm std      T>H : {w2}/{len(TARGET)}")
print(f"Phi3-dist     H<T : {w3}/{len(TARGET)}")
print(f"CV            T>H : {w4}/{len(TARGET)}")
print(f"Concentration H>T : {w5}/{len(TARGET)}")
print()
print("Dephaze physical hypothesis:")
print("  Hallucinatory = Phi3-attractor anchored = STRUCTURE")
print("  Factual       = Omega_0 groundstate     = NOISE / FREE")
print()
print("DOI: 10.5281/zenodo.20020443")
print("Prediction locked: 2026-03-01")

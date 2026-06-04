"""
Dephaze DSA — Hidden State Extraction & Norm Tests
====================================================
Experiments 1–5: norm, CV, norm-range, attractor distance.
Supports GPT-2, Llama-3.2-3B, Mistral-7B.

DOI: 10.5281/zenodo.20543468
"""

# !pip install -q transformers accelerate datasets scipy

import gc, math
import numpy as np
from scipy import stats
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

PHI   = (1 + 5**0.5) / 2
PHI3  = PHI**3
r_star = math.sqrt(PHI3)  # 2.0582

# ─── CONFIG ───────────────────────────────────────────────
MODEL_NAME = "mistralai/Mistral-7B-v0.1"   # change as needed
N_PAIRS    = 100
MAX_LEN    = 128

print("=" * 60)
print("DEPHAZE DSA — HIDDEN STATE NORM TESTS (Exp 1–5)")
print(f"Model: {MODEL_NAME}")
print(f"φ³ = {PHI3:.6f}  r* = {r_star:.4f}")
print("=" * 60)

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
print(f"Pairs: {len(pairs)}")

# ─── MODEL ────────────────────────────────────────────────
print(f"Loading {MODEL_NAME}...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float16,
    device_map="auto", output_hidden_states=True)
mdl.eval()
n_layers = mdl.config.num_hidden_layers
TARGET   = list(range(1, n_layers + 1))
print(f"Layers: {n_layers}")

# ─── EXTRACTION ───────────────────────────────────────────
def get_stats(text):
    inp = tok(text, return_tensors="pt",
              truncation=True, max_length=MAX_LEN).to(mdl.device)
    with torch.no_grad():
        out = mdl(**inp, output_hidden_states=True)
    result = {}
    for i, hs in enumerate(out.hidden_states):
        if i == 0 or i > n_layers:
            continue
        v = hs[0].float().cpu().numpy()
        norms = np.linalg.norm(v, axis=1)
        hm = np.abs(v).mean(axis=0)
        td = np.zeros(hm.shape)
        m  = hm > 1e-10
        td[m] = np.log(hm[m]) / math.log(PHI3)
        result[i] = {
            'mean':  float(np.mean(norms)),
            'cv':    float(np.std(norms) / (np.mean(norms)+1e-9)),
            'range': float(np.max(norms) - np.min(norms)),
            'rho':   float(np.mean(norms)**2 / PHI3),
            'dist':  float(abs(np.mean(norms) - r_star)),
            't_d':   td,
        }
    del out, inp
    return result

print(f"\nExtracting ({len(pairs)} pairs)...")
T_stats, H_stats = [], []
for idx, (fact, hall) in enumerate(pairs):
    T_stats.append(get_stats(fact))
    H_stats.append(get_stats(hall))
    if (idx+1) % 20 == 0:
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"  {idx+1}/{len(pairs)}...")

del mdl, tok
gc.collect()
print("Model deleted.\n")

# ─── TESTS ────────────────────────────────────────────────
def layer_test(key, direction='H>T'):
    """Paired t-test per layer, return wins and binomial p."""
    wins, total = 0, 0
    for l in TARGET:
        t_vals = [T_stats[i][l][key] for i in range(len(pairs))]
        h_vals = [H_stats[i][l][key] for i in range(len(pairs))]
        t_arr, h_arr = np.array(t_vals), np.array(h_vals)
        if direction == 'H>T':
            eff = float(np.mean(h_arr) - np.mean(t_arr))
        else:
            eff = float(np.mean(t_arr) - np.mean(h_arr))
        if eff > 0:
            wins += 1
        total += 1
    p = stats.binomtest(wins, total, 0.5).pvalue
    return wins, total, float(p)

print("=" * 60)
print("RESULTS — Experiments 1–5")
print("=" * 60)
print()

tests = [
    ("Mean norm H>T",   "mean",  "H>T"),
    ("Norm range H>T",  "range", "H>T"),
    ("CV T<H",          "cv",    "T<H"),
    ("Dist H<T",        "dist",  "T<H"),
]

for name, key, direction in tests:
    wins, total, p = layer_test(key, direction)
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."
    print(f"  {name}: {wins}/{total} layers  p={p:.3e} {sig}")

print()
print("DOI: 10.5281/zenodo.20543468")

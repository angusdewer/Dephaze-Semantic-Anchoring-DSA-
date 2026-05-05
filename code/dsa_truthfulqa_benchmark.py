"""
Dephaze Semantic Anchoring (DSA) — TruthfulQA Benchmark
=========================================================
DOI: 10.5281/zenodo.20020443

Experiment 4 from the paper:
  Mistral-7B on TruthfulQA (N=100)
  Mean hidden-state norm H > T: 31/31 layers, p = 0.000

Requires: transformers, torch, accelerate, datasets, scipy, numpy
Hardware: Google Colab T4 GPU (free tier)

Usage:
  python dsa_truthfulqa_benchmark.py
"""

import gc
import torch
import numpy as np
from scipy import stats
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

PHI   = (1 + 5**0.5) / 2
PHI3  = PHI**3
r_star = PHI3**0.5

print(f"phi3 = {PHI3:.6f}  r* = {r_star:.6f}")

# === CLEANUP ===
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
print(f"Pairs loaded: {len(pairs)}")

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
print(f"Ready. dim={mdl.config.hidden_size}, layers={mdl.config.num_hidden_layers}")

# === EXTRACTION ===
def get_layer_stats(text):
    inp = tok(text, return_tensors="pt",
              truncation=True, max_length=128).to(mdl.device)
    with torch.no_grad():
        out = mdl(**inp, output_hidden_states=True)
    result = {}
    for i, hs in enumerate(out.hidden_states):
        v     = hs[0, :, :].float().cpu().numpy()
        norms = np.linalg.norm(v, axis=1)
        result[i] = {
            'mean':  float(np.mean(norms)),
            'cv':    float(np.std(norms) / (np.mean(norms) + 1e-9)),
            'range': float(np.max(norms) - np.min(norms)),
            'dist':  float(np.mean(np.abs(norms - r_star))),
            'std':   float(np.std(norms)),
        }
    del out, inp
    return result

print(f"\nExtracting hidden states ({len(pairs)} pairs)...")
all_t, all_h = [], []
for idx, (fact, hall) in enumerate(pairs):
    all_t.append(get_layer_stats(fact))
    all_h.append(get_layer_stats(hall))
    if (idx+1) % 20 == 0:
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  {idx+1}/{len(pairs)} done...")

del mdl, tok
gc.collect()
torch.cuda.empty_cache()
print("Model deleted, memory freed.\n")

n_layers = len(all_t[0])
n_pairs  = len(pairs)

# === RESULTS ===
def analyse(metric, pred, direction):
    print(f"\n{'='*65}")
    print(f"METRIC: {metric.upper()}")
    print(f"PREDICTION: {pred}")
    print(f"{'='*65}")
    print(f"{'Layer':>6} {'T_mean':>9} {'H_mean':>9} "
          f"{'Diff':>9} {'Correct':>8} {'p':>8} {'sig':>4}")
    print("-"*60)

    wins, sig_wins = 0, 0
    for i in range(1, n_layers - 1):
        t = np.array([d[i][metric] for d in all_t])
        h = np.array([d[i][metric] for d in all_h])
        diff = float(np.mean(t) - np.mean(h))

        if direction == 'H>T':
            correct = diff < 0
        elif direction == 'T>H':
            correct = diff > 0
        else:  # H<T (dist)
            correct = diff > 0

        if correct:
            wins += 1
        _, p = stats.ttest_rel(t, h)
        sig  = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
        if sig and correct:
            sig_wins += 1

        if sig or i % 5 == 0:
            print(f"{i:>6} {np.mean(t):>9.4f} {np.mean(h):>9.4f} "
                  f"{diff:>+9.4f} {str(correct):>8} {p:>8.5f} {sig:>4}"
                  + (" <-- DSA" if sig and correct else ""))

    bp = stats.binomtest(wins, n_layers - 2, 0.5).pvalue
    print(f"\nCorrect: {wins}/{n_layers-2}  "
          f"Sig+correct: {sig_wins}/{n_layers-2}  "
          f"binomial p={bp:.8f}")
    return wins, sig_wins

w1, s1 = analyse('mean',  'Hallucinatory norm LARGER (semantic escape)',    'H>T')
w2, s2 = analyse('cv',    'Factual CV LARGER (noise-like, free)',           'T>H')
w3, s3 = analyse('range', 'Hallucinatory range LARGER (wider spread)',      'H>T')
w4, s4 = analyse('dist',  'Hallucinatory CLOSER to r*=√φ³ (Phi3-attractor)','H<T')

# === SUMMARY ===
print(f"\n{'='*65}")
print("SUMMARY — TruthfulQA Benchmark (N=100) — Mistral-7B")
print(f"{'='*65}")
print(f"phi3 = {PHI3:.6f}   r* = √φ³ = {r_star:.6f}")
print()
print(f"Mean norm  H>T correct: {w1}/{n_layers-2}  sig+correct: {s1}")
print(f"CV         T>H correct: {w2}/{n_layers-2}  sig+correct: {s2}")
print(f"Range      H>T correct: {w3}/{n_layers-2}  sig+correct: {s3}")
print(f"Dist r*    H<T correct: {w4}/{n_layers-2}  sig+correct: {s4}")
print()
print("DSA interpretation:")
print("  Hallucinatory = anchored on Phi3-manifold = STRUCTURE")
print("  Factual       = Omega_0 groundstate       = NOISE / FREE")
print()
print("Prediction locked: 2026-03-01")
print("DOI: 10.5281/zenodo.20020443")

# === SAVE RESULTS ===
with open("results/truthfulqa_results.txt", "w") as f:
    f.write(f"TruthfulQA Benchmark Results\n")
    f.write(f"Model: Mistral-7B-v0.1\n")
    f.write(f"N pairs: {n_pairs}\n")
    f.write(f"phi3 = {PHI3:.10f}\n")
    f.write(f"r*   = {r_star:.10f}\n\n")
    f.write(f"Mean norm  H>T: {w1}/{n_layers-2} layers correct\n")
    f.write(f"CV         T>H: {w2}/{n_layers-2} layers correct\n")
    f.write(f"Range      H>T: {w3}/{n_layers-2} layers correct\n")
    f.write(f"Dist r*    H<T: {w4}/{n_layers-2} layers correct\n")
    f.write(f"\nDOI: 10.5281/zenodo.20020443\n")
    f.write(f"Prediction locked: 2026-03-01\n")
print("\nResults saved to results/truthfulqa_results.txt")

"""
Dephaze DSA — TruthfulQA Benchmark
====================================
Experiments 3–4: CV and norm tests on TruthfulQA N=100.
Reproduces the main results of the paper.

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
r_star = math.sqrt(PHI3)

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
N_PAIRS    = 100

print("=" * 60)
print("DEPHAZE DSA — TruthfulQA BENCHMARK")
print(f"Model: {MODEL_NAME}  N={N_PAIRS}")
print("=" * 60)

# Data
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

# Model
print(f"Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float16,
    device_map="auto", output_hidden_states=True)
mdl.eval()
n_layers = mdl.config.num_hidden_layers
TARGET = list(range(1, n_layers + 1))

def extract(text):
    inp = tok(text, return_tensors="pt",
              truncation=True, max_length=128).to(mdl.device)
    with torch.no_grad():
        out = mdl(**inp, output_hidden_states=True)
    res = {}
    for i, hs in enumerate(out.hidden_states):
        if i == 0 or i > n_layers: continue
        v = hs[0].float().cpu().numpy()
        norms = np.linalg.norm(v, axis=1)
        res[i] = {
            'mean': float(np.mean(norms)),
            'cv':   float(np.std(norms)/(np.mean(norms)+1e-9)),
        }
    del out, inp
    return res

print(f"\nExtracting ({len(pairs)} pairs)...")
T_all, H_all = [], []
for idx, (fact, hall) in enumerate(pairs):
    T_all.append(extract(fact))
    H_all.append(extract(hall))
    if (idx+1) % 20 == 0:
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"  {idx+1}/{len(pairs)}...")

del mdl, tok
gc.collect()
print("Model deleted.\n")

# Results
print("=" * 60)
print("LAYER-BY-LAYER RESULTS")
print("=" * 60)
print(f"\n{'Layer':>6} {'T_mean':>9} {'H_mean':>9} {'H>T':>5} "
      f"{'T_cv':>9} {'H_cv':>9} {'T<H':>5}")
print("-" * 60)

mean_wins = cv_wins = 0

for l in TARGET:
    t_mean = np.array([T_all[i][l]['mean'] for i in range(len(pairs))])
    h_mean = np.array([H_all[i][l]['mean'] for i in range(len(pairs))])
    t_cv   = np.array([T_all[i][l]['cv']   for i in range(len(pairs))])
    h_cv   = np.array([H_all[i][l]['cv']   for i in range(len(pairs))])

    mean_H_gt_T = np.mean(h_mean) > np.mean(t_mean)
    cv_T_lt_H   = np.mean(t_cv)   < np.mean(h_cv)
    if mean_H_gt_T: mean_wins += 1
    if cv_T_lt_H:   cv_wins   += 1

    print(f"{l:>6} {np.mean(t_mean):>9.3f} {np.mean(h_mean):>9.3f} "
          f"{'✓' if mean_H_gt_T else '':>5} "
          f"{np.mean(t_cv):>9.4f} {np.mean(h_cv):>9.4f} "
          f"{'✓' if cv_T_lt_H else '':>5}")

p_mean = stats.binomtest(mean_wins, len(TARGET), 0.5).pvalue
p_cv   = stats.binomtest(cv_wins,   len(TARGET), 0.5).pvalue

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nMean norm H>T: {mean_wins}/{len(TARGET)} layers  p={p_mean:.3e}")
print(f"CV T<H:        {cv_wins}/{len(TARGET)} layers  p={p_cv:.3e}")
print()
print("Expected (paper): 31/31, p=0.000")
print()
print("DOI: 10.5281/zenodo.20543468")

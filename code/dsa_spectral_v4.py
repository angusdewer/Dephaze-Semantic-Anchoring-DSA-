"""
Dephaze DSA — Spektrális Elemzés v4 (dispatch fix)
====================================================
Meta device fix: rétegenkénti explicit GPU dispatch
DOI: 10.5281/zenodo.20543468
"""

# !pip install -q transformers accelerate scikit-learn

import gc, math
import numpy as np
from sklearn.utils.extmath import randomized_svd
import torch
from transformers import AutoModelForCausalLM

PHI  = (1+5**0.5)/2
PHI3 = PHI**3
D_MIN = 4096
fib_idx = {n: round(D_MIN * PHI**(-n)) for n in range(1, 8)}
TEST_LAYERS = [1, 8, 12, 20, 24, 28]
N_SV = 150

print("=" * 68)
print("SPEKTRÁLIS ELEMZÉS v4 — dispatch fix")
print(f"φ={PHI:.6f}  φ³−φ⁻³={PHI3-1/PHI3:.6f}")
print("=" * 68)

def analyze_weight(W_np, name, n_sv=N_SV):
    rows, cols = W_np.shape
    n_comp = min(n_sv, min(rows, cols) - 1)
    _, S, _ = randomized_svd(W_np, n_components=n_comp, random_state=42)
    S = np.sort(S)[::-1]
    ratios = S[:-1] / (S[1:] + 1e-12)
    r_mean = float(np.mean(ratios))

    # Véletlen proxy
    pr, pc = min(rows, 500), min(cols, 500)
    W_rand = np.random.randn(pr, pc).astype(np.float32)
    W_rand *= np.std(W_np) / (np.std(W_rand) + 1e-12)
    _, Sr, _ = randomized_svd(W_rand,
        n_components=min(n_comp, pr-1, pc-1), random_state=0)
    Sr = np.sort(Sr)[::-1]
    r_rand = float(np.mean(Sr[:-1] / (Sr[1:] + 1e-12)))

    h1 = abs(r_mean - PHI) < abs(r_rand - PHI)

    # Fibonacci töréspontok
    fib_matches = 0
    if len(S) > 10:
        log_s = np.log(S + 1e-12)
        log_i = np.log(np.arange(1, len(S)+1))
        d2 = np.abs(np.gradient(np.gradient(log_s, log_i), log_i))
        top_breaks = np.argsort(d2)[::-1][:8] + 1
        fib_matches = sum(
            1 for bp in top_breaks
            for fi in fib_idx.values()
            if fi <= n_sv and abs(bp-fi) <= max(3, fi*0.08)
        )

    print(f"    {name:>12} {(rows,cols)}  "
          f"r={r_mean:.4f}  |r-φ|={abs(r_mean-PHI):.4f}  "
          f"ctrl={abs(r_rand-PHI):.4f}  "
          f"H1:{'✓' if h1 else '✗'}  Fib:{fib_matches}")
    return {"r_mean": r_mean, "r_rand": r_rand,
            "h1": h1, "fib_matches": fib_matches}

# ─── BETÖLTÉS ─────────────────────────────────────────────
print("\nBetöltés (float16)...")
mdl = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
mdl.eval()

# GPU eszköz meghatározása
device = next(
    (p.device for p in mdl.parameters() if p.device.type != 'meta'),
    torch.device('cpu')
)
print(f"Aktív eszköz: {device}\n")

all_results = []

for li in TEST_LAYERS:
    print(f"Réteg {li}:")
    layer = mdl.model.layers[li]

    results_layer = {}
    for proj_name in ["gate_proj", "down_proj"]:
        proj = getattr(layer.mlp, proj_name)

        # Meta device fix: dispatch a réteg egészét GPU-ra
        if proj.weight.device.type == 'meta':
            try:
                layer = layer.to(device)
                proj  = getattr(layer.mlp, proj_name)
            except Exception as e:
                print(f"    {proj_name}: dispatch hiba ({e})")
                continue

        try:
            W_np = proj.weight.detach().float().cpu().numpy()
            r = analyze_weight(W_np, proj_name)
            results_layer[proj_name] = r
            del W_np
        except Exception as e:
            print(f"    {proj_name}: hiba ({e})")

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if "gate_proj" in results_layer and "down_proj" in results_layer:
        all_results.append({
            "layer": li,
            "gate": results_layer["gate_proj"],
            "down": results_layer["down_proj"],
        })

del mdl
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ─── ÖSSZEFOGLALÁS ────────────────────────────────────────
print()
print("=" * 68)
print("ÖSSZEFOGLALÁS")
print("=" * 68)
print(f"\n{'Réteg':>6} {'gate_r':>8} {'H1_g':>5} "
      f"{'down_r':>8} {'H1_d':>5} {'Fib':>5}")
print("-" * 42)

h1g = h1d = 0
for r in all_results:
    g, d = r["gate"], r["down"]
    if g["h1"]: h1g += 1
    if d["h1"]: h1d += 1
    fm = g["fib_matches"] + d["fib_matches"]
    print(f"{r['layer']:>6} {g['r_mean']:>8.4f} "
          f"{'✓' if g['h1'] else '✗':>5} "
          f"{d['r_mean']:>8.4f} "
          f"{'✓' if d['h1'] else '✗':>5} "
          f"{fm:>5}")

n = len(all_results)
print()
all_r = ([r["gate"]["r_mean"] for r in all_results] +
         [r["down"]["r_mean"] for r in all_results])
all_rand = ([r["gate"]["r_rand"] for r in all_results] +
            [r["down"]["r_rand"] for r in all_results])

print(f"Valódi mátrix  σᵢ/σᵢ₊₁ átlag: {np.mean(all_r):.4f}")
print(f"Véletlen ctrl  σᵢ/σᵢ₊₁ átlag: {np.mean(all_rand):.4f}")
print(f"φ                             : {PHI:.4f}")
print()
print(f"H1 gate_proj: {h1g}/{n} rétegben")
print(f"H1 down_proj: {h1d}/{n} rétegben")
print()

if n == 0:
    print("HIBA: Egy réteg sem futott le.")
elif h1g >= n*0.6 or h1d >= n*0.6:
    print("SPEKTRÁLIS IGAZOLÁS ✓")
    print(f"  {h1g+h1d}/{2*n} esetben a valódi mátrix közelebb φ-hoz")
    print("  mint véletlen baseline.")
    print("  Az algebrai lánc zárható:")
    print("  ff/d=φ³−φ⁻³=4 → W spektrum φ-szervezett")
    print("  → hidden state φ-Fibonacci struktúra (szükséges és elégséges)")
else:
    print("KONZERVATÍV ÁLLÍTÁS:")
    print("  ff/d=4=φ³−φ⁻³ szükséges de nem elégséges.")
    print("  Formális spektrális zárás: jövő munka.")

print("\nDOI: 10.5281/zenodo.20543468")

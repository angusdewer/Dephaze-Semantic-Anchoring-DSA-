"""
Dephaze DSA — Log-Egyenletes Kontroll Teszt
=============================================
REVIEWER KÉRDÉS:
  A Fibonacci lépések (369, 598, 967, 1565) logaritmikusan
  egyenletesek — φ alapon. Ez önmagában adhat struktúrát
  függetlenül φ-tól.

TESZT:
  Összehasonlítjuk φ alapú és más bázisú log-egyenletes lépéseket.

  φ  alapú: k_n = round(4096 × φ^{-n}),  n=5,4,3,2
            → 369, 598, 967, 1565
            → arány: φ = 1.6180

  e  alapú: k_n = round(4096 × e^{-n}),  n=5,4,3,2
            → 55, 150, 409, 1112
            → arány: e = 2.7183

  2  alapú: k_n = round(4096 × 2^{-n}),  n=5,4,3,2
            → 128, 256, 512, 1024
            → arány: 2

  1.5 alapú: k_n = round(4096 × 1.5^{-n}), n=5,4,3,2
            → arány: 1.5

  √2 alapú: k_n = round(4096 × √2^{-n}),  n=5,4,3,2
            → arány: √2 = 1.4142

  Mind logaritmikusan egyenletes — csak a bázis más.
  Ha φ kiemelkedik ezek közül → φ-SPECIFIKUS, nem csak log-egyenletes.

DOI: 10.5281/zenodo.20543468
"""

# !pip install -q transformers accelerate datasets scipy

import gc
import math
import numpy as np
from scipy import stats
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

PHI   = (1 + 5**0.5) / 2
PHI3  = PHI**3
LN3   = math.log(PHI3)
N_LAYERS = 32
D = 4096

# Log-egyenletes lépés-négyesek különböző bázisokkal
# Minden esetben: [k_a, k_b] lenyomat + [k_c, k_d] generáló
# ugyanolyan struktúra mint Φ³_ext

def make_log_steps(base, n_list, D=4096):
    """k_n = round(D × base^{-n}) az n_list értékeire."""
    steps = [round(D * base**(-n)) for n in n_list]
    return [max(10, min(s, D-10)) for s in steps]

# n értékek: 5,4,3,2 — ugyanaz mint a Fibonacci esetben
N_LIST = [5, 4, 3, 2]

BASES = {
    f"φ  = {PHI:.4f}  [Fibonacci]": PHI,
    f"√2 = {math.sqrt(2):.4f}": math.sqrt(2),
    f"1.5= 1.5000": 1.5,
    f"e  = {math.e:.4f}": math.e,
    f"2  = 2.0000": 2.0,
    f"3  = 3.0000": 3.0,
    f"√3 = {math.sqrt(3):.4f}": math.sqrt(3),
    f"φ² = {PHI**2:.4f}": PHI**2,
}

print("=" * 68)
print("DEPHAZE DSA — LOG-EGYENLETES KONTROLL TESZT")
print("φ alapú vs más bázisú logaritmikusan egyenletes lépések")
print("=" * 68)
print()
print("Lépés-négyesek (n=5,4,3,2):")
print(f"{'Bázis':>20} {'k_a':>6} {'k_b':>6} {'k_c':>6} {'k_d':>6}  arány")
print("-" * 55)
for name, base in BASES.items():
    steps = make_log_steps(base, N_LIST)
    ratios = [steps[i+1]/steps[i] for i in range(len(steps)-1)]
    print(f"{name:>20} {steps[0]:>6} {steps[1]:>6} "
          f"{steps[2]:>6} {steps[3]:>6}  "
          f"{ratios[0]:.3f}, {ratios[1]:.3f}, {ratios[2]:.3f}")
print()

TARGET = list(range(1, N_LAYERS + 1))
STRONG = list(range(3, 32))

# ─── ADATOK ───────────────────────────────────────────────────
print("TruthfulQA betöltése...")
ds = load_dataset("truthfulqa/truthful_qa", "generation")
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
print(f"Párok: {len(pairs)}")

# ─── MODEL ────────────────────────────────────────────────────
print("Mistral-7B betöltése...")
tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
mdl = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    dtype=torch.float16,
    device_map="auto",
    output_hidden_states=True
)
mdl.eval()
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
print(f"Kész.")

# ─── KINYERÉS ─────────────────────────────────────────────────
def get_t_vectors(text):
    inp = tok(text, return_tensors="pt",
              truncation=True, max_length=128).to(mdl.device)
    with torch.no_grad():
        out = mdl(**inp, output_hidden_states=True)
    result = {}
    for i in TARGET:
        h = out.hidden_states[i][0].float().cpu().numpy()
        h_mean = np.mean(np.abs(h), axis=0)
        valid = h_mean > 1e-10
        t_d = np.zeros(h_mean.shape)
        t_d[valid] = np.log(h_mean[valid]) / LN3
        result[i] = t_d
    del out, inp
    return result

print(f"\nKinyerés ({len(pairs)} pár)...")
all_t_vecs = []
all_h_vecs = []
for idx_p, (fact, hall) in enumerate(pairs):
    all_t_vecs.append(get_t_vectors(fact))
    all_h_vecs.append(get_t_vectors(hall))
    if (idx_p + 1) % 20 == 0:
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"  {idx_p+1}/{len(pairs)}...")

del mdl, tok
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
print("Model törölve.\n")

# ─── MÉRŐSZÁM ─────────────────────────────────────────────────
def general_index(tv, ka, kb, kc, kd):
    """[corr(ka)+corr(kb)] - [corr(kc)+corr(kd)]"""
    def c(k):
        if len(tv) <= k or k <= 0:
            return 0.0
        a, b = tv[:-k], tv[k:]
        valid = (np.abs(a) > 1e-10) & (np.abs(b) > 1e-10)
        if valid.sum() < 10:
            return 0.0
        r, _ = stats.pearsonr(a[valid], b[valid])
        return float(r)
    return (c(ka) + c(kb)) - (c(kc) + c(kd))

def compute_base_effect(steps, layers):
    """
    T-H effect és réteg-szintű wins egy lépés-négyesre.
    steps = [ka, kb, kc, kd]
    """
    ka, kb, kc, kd = steps
    layer_wins = 0
    layer_effs = []
    for l in layers:
        t_idx = [general_index(all_t_vecs[i][l], ka, kb, kc, kd)
                 for i in range(len(pairs))]
        h_idx = [general_index(all_h_vecs[i][l], ka, kb, kc, kd)
                 for i in range(len(pairs))]
        eff = float(np.mean(np.array(t_idx)) - np.mean(np.array(h_idx)))
        layer_effs.append(eff)
        if eff > 0:
            layer_wins += 1
    p_b = stats.binomtest(layer_wins, len(layers), 0.5).pvalue
    mean_eff = float(np.mean(layer_effs))
    return layer_wins, p_b, mean_eff, layer_effs

# ═══════════════════════════════════════════════════════════════
# FŐTESZT: minden bázis réteg-szintű wins és effect
# ═══════════════════════════════════════════════════════════════

print("=" * 68)
print("LOG-EGYENLETES BÁZIS ÖSSZEHASONLÍTÁS")
print("Minden bázis: [corr(k_a)+corr(k_b)] - [corr(k_c)+corr(k_d)]")
print("n=5,4 → lenyomat ág;  n=3,2 → generáló ág")
print("=" * 68)
print()
print(f"{'Bázis':>22} {'ka,kb,kc,kd':>28} "
      f"{'wins/32':>8} {'p_binom':>10} {'effect':>10}")
print("-" * 82)

results = {}
phi_effect = None
phi_wins = None

for name, base in BASES.items():
    steps = make_log_steps(base, N_LIST)
    ka, kb, kc, kd = steps
    wins, p_b, mean_eff, layer_effs = compute_base_effect(steps, TARGET)
    results[name] = (wins, p_b, mean_eff, layer_effs, steps)

    marker = "  ← FIBONACCI (referencia)" if "Fibonacci" in name else ""
    if "Fibonacci" in name:
        phi_effect = mean_eff
        phi_wins = wins

    sig = "***" if p_b<0.001 else "**" if p_b<0.01 else \
          "*" if p_b<0.05 else ""
    print(f"{name:>22} ({ka},{kb},{kc},{kd})"
          f"{wins:>5}/32 {p_b:>10.2e}{sig:>4} {mean_eff:>+10.6f}{marker}")

# ═══════════════════════════════════════════════════════════════
# RÉTEGENKÉNTI BONTÁS — φ vs legjobb kontroll
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*68}")
print("RÉTEGENKÉNTI BONTÁS — φ vs legjobb nem-φ bázis")
print(f"{'='*68}\n")

# Legjobb kontroll meghatározása
best_ctrl_name = max(
    [n for n in results if "Fibonacci" not in n],
    key=lambda n: results[n][2]
)
best_ctrl = results[best_ctrl_name]

print(f"Legjobb kontroll: {best_ctrl_name}")
print(f"  Steps: {best_ctrl[4]}")
print(f"  Effect: {best_ctrl[2]:+.6f}  wins: {best_ctrl[0]}/32")
print()

phi_effs = results[[n for n in results if "Fibonacci" in n][0]][3]
ctrl_effs = best_ctrl[3]

print(f"{'Réteg':>6} {'φ effect':>10} {'ctrl effect':>12} "
      f"{'φ-ctrl':>10}  φ jobb?")
print("-" * 50)

phi_better_count = 0
for l_idx, l in enumerate(TARGET):
    phi_e = phi_effs[l_idx]
    ctrl_e = ctrl_effs[l_idx]
    diff = phi_e - ctrl_e
    better = "✓" if diff > 0 else ""
    if diff > 0:
        phi_better_count += 1

    zone = ""
    for b, bn in [(8,"φ⁻³"), (12,"φ⁻²"), (20,"φ⁻¹"), (24,"1-φ⁻³")]:
        if abs(l-b) <= 1:
            zone = f"  ← {bn}"

    print(f"{l:>6} {phi_e:>+10.6f} {ctrl_e:>+12.6f} "
          f"{diff:>+10.6f}  {better}{zone}")

p_phi_better = stats.binomtest(phi_better_count, len(TARGET), 0.5).pvalue
print(f"\nφ jobb mint legjobb kontroll: {phi_better_count}/{len(TARGET)} rétegben")
print(f"Binomiális p = {p_phi_better:.6f}")

# ═══════════════════════════════════════════════════════════════
# STATISZTIKAI ÖSSZEHASONLÍTÁS
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*68}")
print("STATISZTIKAI ÖSSZEHASONLÍTÁS")
print(f"{'='*68}\n")

ctrl_effects = [results[n][2] for n in results if "Fibonacci" not in n]
ctrl_wins_list = [results[n][0] for n in results if "Fibonacci" not in n]

print(f"φ Fibonacci:      wins={phi_wins}/32  effect={phi_effect:+.6f}")
print(f"Kontrollok átlag: wins={np.mean(ctrl_wins_list):.1f}/32  "
      f"effect={np.mean(ctrl_effects):+.6f}")
print(f"Kontrollok std:   wins={np.std(ctrl_wins_list):.1f}     "
      f"effect={np.std(ctrl_effects):.6f}")

z_wins = (phi_wins - np.mean(ctrl_wins_list)) / (np.std(ctrl_wins_list) + 1e-10)
z_eff  = (phi_effect - np.mean(ctrl_effects)) / (np.std(ctrl_effects) + 1e-10)
p_better = np.mean(np.array(ctrl_effects) >= phi_effect)

print(f"\nZ-score (wins):   {z_wins:+.2f}")
print(f"Z-score (effect): {z_eff:+.2f}")
print(f"P(log-egyenletes kontroll ≥ φ): {p_better:.4f}")
print()

if p_better < 0.05:
    print("DÖNTÉS: φ SPECIFIKUS a log-egyenletes struktúrán belül is ✓")
    print("Nem elég logaritmikusan egyenletesnek lenni — φ bázis különleges.")
elif p_better < 0.15:
    print("DÖNTÉS: HATÁRESET — gyenge φ specificitás a log-egyenletesen belül")
else:
    print("DÖNTÉS: NEM SPECIFIKUS a log-egyenletesen belül")
    print("A log-egyenletes eloszlás elegendő — φ nem különleges.")

# ═══════════════════════════════════════════════════════════════
# ÖSSZEFOGLALÁS
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*68}")
print("ÖSSZEFOGLALÁS")
print(f"{'='*68}")
print()
print("Kérdés: a Φ³_ext eredmény φ-specifikus,")
print("vagy elég logaritmikusan egyenletesnek lenni?")
print()
print("Tesztelt bázisok (mind log-egyenletes, n=5,4,3,2):")
for name, (wins, p_b, eff, _, steps) in results.items():
    marker = " ← REFERENCIA" if "Fibonacci" in name else ""
    print(f"  {name:>22}: wins={wins}/32  effect={eff:+.6f}{marker}")
print()
print(f"Z-score (effect): {z_eff:+.2f}")
print(f"P(ctrl ≥ φ):      {p_better:.4f}")
print()
print("Reviewer kérdés megválaszolva:")
print("  Ha p < 0.05: φ specifikus a log-egyenletesen belül is")
print("  Ha p ≥ 0.05: log-egyenletes struktúra elég, φ nem különleges")
print()
print("DOI: 10.5281/zenodo.20543468")

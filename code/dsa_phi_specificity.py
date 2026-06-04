"""
Dephaze DSA — φ Specificitás Teszt (Döntő)
===========================================
KRITIKUS KÉRDÉS:
  A Φ³_ext eredmény (31/32, p=1.5×10⁻⁸) φ-specifikus?
  Vagy bármely k értéknél ugyanolyan T-H különbség látható?

TESZT DESIGN:
  Φ³_ext = [corr(k5=369)+corr(k4=598)] - [corr(k3=967)+corr(k2=1565)]

  Kontroll: ugyanolyan struktúrájú index más k értékekkel
  Φ_ctrl  = [corr(k_a)+corr(k_b)] - [corr(k_c)+corr(k_d)]

  ahol k_a, k_b, k_c, k_d NEM Fibonacci lépések

  Ha Φ³_ext T-H effect >> Φ_ctrl T-H effect → φ SPECIFIKUS
  Ha közel egyenlő → általános autokorreláció, φ nem különleges

KONTROLL STRATÉGIÁK:
  1. Eltolt lépések: k±50 (Fibonacci közelében de nem azon)
  2. Véletlen lépések: egyenletes eloszlásból
  3. Geometriai lépések más alappal: √2, e, π alapú lépések
  4. Lineáris lépések: egyenletes rács

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

# Fibonacci lépések
k5 = round(D * PHI**(-5))  # 369
k4 = round(D * PHI**(-4))  # 598
k3 = round(D * PHI**(-3))  # 967
k2 = round(D * PHI**(-2))  # 1565

print("=" * 68)
print("DEPHAZE DSA — φ SPECIFICITÁS TESZT (DÖNTŐ)")
print("Fibonacci lépések vs kontroll lépések")
print("=" * 68)
print(f"\nFibonacci lépések:")
print(f"  k5={k5}, k4={k4}, k3={k3}, k2={k2}")
print(f"  k3/k4 = {k3/k4:.6f} ≈ φ = {PHI:.6f}")
print()

# Kontroll lépés-csoportok
# Minden csoport: [k_a, k_b] lenyomat + [k_c, k_d] generáló
# Ugyanolyan struktúra mint Φ³_ext

# Eltolt kontrollok (±Δ a Fibonacci értékektől)
DELTAS = [25, 50, 75, 100, 150]
ctrl_shifted = {}
for d in DELTAS:
    ctrl_shifted[f"+{d}"] = (k5+d, k4+d, k3+d, k2+d)
    ctrl_shifted[f"-{d}"] = (k5-d, k4-d, k3-d, k2-d)

# Véletlen lépés-csoportok (seed rögzítve)
np.random.seed(42)
N_RANDOM = 20
ctrl_random = {}
for i in range(N_RANDOM):
    # 4 véletlen k érték 200-1800 között
    ks = sorted(np.random.randint(200, 1800, 4))
    ctrl_random[f"rand_{i}"] = tuple(ks)

# Más geometriai alapok
BASES = {
    "√2": math.sqrt(2),
    "e":  math.e,
    "π":  math.pi,
    "3/2":1.5,
    "√3": math.sqrt(3),
}
ctrl_geometric = {}
for base_name, base in BASES.items():
    # Hasonló struktúra: 4 lépés a base hatványsorozatán
    # k_a < k_b < k_c < k_d, arány ≈ base
    k_start = round(D * base**(-5))
    steps = [round(D * base**(-n)) for n in range(5, 1, -1)]
    steps = [k for k in steps if 50 < k < D//2]
    if len(steps) >= 4:
        ctrl_geometric[base_name] = tuple(steps[:4])
    elif len(steps) == 3:
        # Ha csak 3, duplikálunk egyet
        ctrl_geometric[base_name] = (steps[0], steps[1], steps[1], steps[2])

# Lineáris rács
ctrl_linear = {}
for spacing in [200, 250, 300, 350, 400]:
    base_l = 400
    ctrl_linear[f"lin_{spacing}"] = (
        base_l, base_l+spacing,
        base_l+2*spacing, base_l+3*spacing
    )

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
    """
    Általános index: [corr(ka)+corr(kb)] - [corr(kc)+corr(kd)]
    Fibonacci esetén: ka=k5, kb=k4, kc=k3, kd=k2
    """
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

def compute_index_effect(ka, kb, kc, kd, layers):
    """T-H effect és p-érték egy index konfigurációra."""
    t_all, h_all = [], []
    for l in layers:
        for i in range(len(pairs)):
            idx_t = general_index(all_t_vecs[i][l], ka, kb, kc, kd)
            idx_h = general_index(all_h_vecs[i][l], ka, kb, kc, kd)
            t_all.append(idx_t)
            h_all.append(idx_h)
    t_arr = np.array(t_all)
    h_arr = np.array(h_all)
    n = min(len(t_arr), len(h_arr))
    effect = float(np.mean(t_arr[:n]) - np.mean(h_arr[:n]))
    _, p = stats.ttest_rel(t_arr[:n], h_arr[:n])
    wins = int(np.sum(
        np.array([general_index(all_t_vecs[i][l], ka, kb, kc, kd)
                  for l in TARGET for i in range(len(pairs))]) >
        np.array([general_index(all_h_vecs[i][l], ka, kb, kc, kd)
                  for l in TARGET for i in range(len(pairs))])
    ))
    total = len(TARGET) * len(pairs)
    return effect, float(p), wins, total

# ═══════════════════════════════════════════════════════════════
# REFERENCIA: Fibonacci Φ³_ext
# ═══════════════════════════════════════════════════════════════
print("=" * 68)
print("REFERENCIA: Fibonacci Φ³_ext")
print(f"  [corr({k5})+corr({k4})] - [corr({k3})+corr({k2})]")
print("=" * 68)

# Rétegenkénti wins
phi3_layer_wins = 0
phi3_results = []
for l in TARGET:
    t_idx = [general_index(all_t_vecs[i][l], k5, k4, k3, k2)
             for i in range(len(pairs))]
    h_idx = [general_index(all_h_vecs[i][l], k5, k4, k3, k2)
             for i in range(len(pairs))]
    t_arr = np.array(t_idx)
    h_arr = np.array(h_idx)
    effect = float(np.mean(t_arr) - np.mean(h_arr))
    _, p = stats.ttest_rel(t_arr, h_arr)
    phi3_results.append(effect)
    if effect > 0:
        phi3_layer_wins += 1

phi3_p = stats.binomtest(phi3_layer_wins, len(TARGET), 0.5).pvalue
phi3_mean_eff = np.mean(phi3_results)
print(f"Réteg-szintű wins: {phi3_layer_wins}/32")
print(f"Binomiális p:      {phi3_p:.2e}")
print(f"Átlag effect:      {phi3_mean_eff:+.6f}")
print()

# ═══════════════════════════════════════════════════════════════
# 1. ELTOLT KONTROLLOK
# ═══════════════════════════════════════════════════════════════
print("=" * 68)
print("1. ELTOLT KONTROLLOK (Fibonacci ± Δ)")
print(f"{'Kontroll':>12} {'ka,kb,kc,kd':>28} "
      f"{'wins/32':>8} {'p_binom':>10} {'mean_eff':>10}  vs φ³")
print("-" * 78)

shifted_effects = []
for name, (ka, kb, kc, kd) in ctrl_shifted.items():
    if any(k <= 0 or k >= D for k in [ka, kb, kc, kd]):
        continue
    layer_wins = 0
    layer_effs = []
    for l in TARGET:
        t_idx = [general_index(all_t_vecs[i][l], ka, kb, kc, kd)
                 for i in range(len(pairs))]
        h_idx = [general_index(all_h_vecs[i][l], ka, kb, kc, kd)
                 for i in range(len(pairs))]
        eff = float(np.mean(np.array(t_idx)) - np.mean(np.array(h_idx)))
        layer_effs.append(eff)
        if eff > 0:
            layer_wins += 1
    p_b = stats.binomtest(layer_wins, len(TARGET), 0.5).pvalue
    mean_eff = np.mean(layer_effs)
    shifted_effects.append(mean_eff)
    marker = "✓ hasonló" if layer_wins >= 26 else "✗ gyengébb"
    print(f"{name:>12} ({ka},{kb},{kc},{kd})"
          f"{layer_wins:>5}/32 {p_b:>10.2e} {mean_eff:>+10.6f}  {marker}")

print(f"\nFibonacci átlag effect: {phi3_mean_eff:+.6f}")
print(f"Eltolt átlag effect:    {np.mean(shifted_effects):+.6f}")
print(f"φ³ specifikus vs eltolt: "
      f"{'IGEN ✓' if phi3_mean_eff > np.mean(shifted_effects) + abs(np.std(shifted_effects)) else 'NEM'}")

# ═══════════════════════════════════════════════════════════════
# 2. VÉLETLEN KONTROLLOK
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*68}")
print("2. VÉLETLEN KONTROLLOK (N=20 véletlen k-négyes)")
print(f"{'Kontroll':>12} {'ka,kb,kc,kd':>28} "
      f"{'wins/32':>8} {'p_binom':>10} {'mean_eff':>10}")
print("-" * 72)

random_wins_list = []
random_effects = []
for name, (ka, kb, kc, kd) in ctrl_random.items():
    layer_wins = 0
    layer_effs = []
    for l in TARGET:
        t_idx = [general_index(all_t_vecs[i][l], ka, kb, kc, kd)
                 for i in range(len(pairs))]
        h_idx = [general_index(all_h_vecs[i][l], ka, kb, kc, kd)
                 for i in range(len(pairs))]
        eff = float(np.mean(np.array(t_idx)) - np.mean(np.array(h_idx)))
        layer_effs.append(eff)
        if eff > 0:
            layer_wins += 1
    p_b = stats.binomtest(layer_wins, len(TARGET), 0.5).pvalue
    mean_eff = np.mean(layer_effs)
    random_wins_list.append(layer_wins)
    random_effects.append(mean_eff)
    print(f"{name:>12} ({ka},{kb},{kc},{kd})"
          f"{layer_wins:>5}/32 {p_b:>10.2e} {mean_eff:>+10.6f}")

print(f"\nFibonacci:        wins={phi3_layer_wins}/32  "
      f"mean_eff={phi3_mean_eff:+.6f}")
print(f"Véletlen átlag:   wins={np.mean(random_wins_list):.1f}/32  "
      f"mean_eff={np.mean(random_effects):+.6f}")
print(f"Véletlen std:     wins={np.std(random_wins_list):.1f}     "
      f"mean_eff={np.std(random_effects):.6f}")

# Z-score: φ³ mennyire kiemelkedik a véletlen eloszlásból
z_wins = (phi3_layer_wins - np.mean(random_wins_list)) / \
         (np.std(random_wins_list) + 1e-10)
z_eff  = (phi3_mean_eff - np.mean(random_effects)) / \
         (np.std(random_effects) + 1e-10)
p_better = np.mean(np.array(random_effects) >= phi3_mean_eff)

print(f"\nZ-score (wins):   {z_wins:+.2f}")
print(f"Z-score (effect): {z_eff:+.2f}")
print(f"P(véletlen ≥ φ³): {p_better:.3f}")
print(f"φ³ SPECIFIKUS: "
      f"{'IGEN ✓' if p_better < 0.05 else 'NEM — általános autokorreláció'}")

# ═══════════════════════════════════════════════════════════════
# 3. MÁS GEOMETRIAI ALAPOK
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*68}")
print("3. MÁS GEOMETRIAI ALAPOK (√2, e, π, 3/2, √3)")
print(f"{'Alap':>8} {'ka,kb,kc,kd':>28} "
      f"{'wins/32':>8} {'p_binom':>10} {'mean_eff':>10}")
print("-" * 68)

geo_effects = []
for base_name, (ka, kb, kc, kd) in ctrl_geometric.items():
    layer_wins = 0
    layer_effs = []
    for l in TARGET:
        t_idx = [general_index(all_t_vecs[i][l], ka, kb, kc, kd)
                 for i in range(len(pairs))]
        h_idx = [general_index(all_h_vecs[i][l], ka, kb, kc, kd)
                 for i in range(len(pairs))]
        eff = float(np.mean(np.array(t_idx)) - np.mean(np.array(h_idx)))
        layer_effs.append(eff)
        if eff > 0:
            layer_wins += 1
    p_b = stats.binomtest(layer_wins, len(TARGET), 0.5).pvalue
    mean_eff = np.mean(layer_effs)
    geo_effects.append(mean_eff)
    marker = "✓ hasonló" if layer_wins >= 26 else ""
    print(f"{base_name:>8} ({ka},{kb},{kc},{kd})"
          f"{layer_wins:>5}/32 {p_b:>10.2e} {mean_eff:>+10.6f}  {marker}")

print(f"\nFibonacci:       {phi3_mean_eff:+.6f}")
print(f"Geo átlag:       {np.mean(geo_effects):+.6f}")
print(f"φ³ specifikus vs geometriai: "
      f"{'IGEN ✓' if phi3_mean_eff > np.mean(geo_effects) + np.std(geo_effects) else 'NEM'}")

# ═══════════════════════════════════════════════════════════════
# 4. LINEÁRIS RÁCS KONTROLL
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*68}")
print("4. LINEÁRIS RÁCS KONTROLL")
print(f"{'Kontroll':>12} {'ka,kb,kc,kd':>28} "
      f"{'wins/32':>8} {'mean_eff':>10}")
print("-" * 65)

lin_effects = []
for name, (ka, kb, kc, kd) in ctrl_linear.items():
    layer_wins = 0
    layer_effs = []
    for l in TARGET:
        t_idx = [general_index(all_t_vecs[i][l], ka, kb, kc, kd)
                 for i in range(len(pairs))]
        h_idx = [general_index(all_h_vecs[i][l], ka, kb, kc, kd)
                 for i in range(len(pairs))]
        eff = float(np.mean(np.array(t_idx)) - np.mean(np.array(h_idx)))
        layer_effs.append(eff)
        if eff > 0:
            layer_wins += 1
    mean_eff = np.mean(layer_effs)
    lin_effects.append(mean_eff)
    print(f"{name:>12} ({ka},{kb},{kc},{kd})"
          f"{layer_wins:>5}/32 {mean_eff:>+10.6f}")

print(f"\nFibonacci:      {phi3_mean_eff:+.6f}")
print(f"Lineáris átlag: {np.mean(lin_effects):+.6f}")

# ─── VÉGSŐ ÍTÉLET ─────────────────────────────────────────────
print(f"\n{'='*68}")
print("VÉGSŐ ÍTÉLET — φ SPECIFICITÁS")
print(f"{'='*68}")
print()
print(f"Fibonacci Φ³_ext:    wins={phi3_layer_wins}/32  "
      f"effect={phi3_mean_eff:+.6f}")
print(f"Eltolt átlag:        effect={np.mean(shifted_effects):+.6f}")
print(f"Véletlen átlag:      effect={np.mean(random_effects):+.6f}")
print(f"Geometriai átlag:    effect={np.mean(geo_effects):+.6f}")
print(f"Lineáris átlag:      effect={np.mean(lin_effects):+.6f}")
print()
all_ctrl = shifted_effects + random_effects + geo_effects + lin_effects
p_final = np.mean(np.array(all_ctrl) >= phi3_mean_eff)
print(f"P(bármely kontroll ≥ Fibonacci): {p_final:.4f}")
print()
if p_final < 0.05:
    print("DÖNTÉS: φ SPECIFIKUS ✓")
    print("A Fibonacci lépések valóban kiemelkednek — nem általános autokorreláció")
elif p_final < 0.15:
    print("DÖNTÉS: HATÁRESET — gyenge φ specificitás")
else:
    print("DÖNTÉS: NEM SPECIFIKUS")
    print("Az eredmény általános autokorreláció struktúra, φ nem különleges")
print()
print("DOI: 10.5281/zenodo.20543468")

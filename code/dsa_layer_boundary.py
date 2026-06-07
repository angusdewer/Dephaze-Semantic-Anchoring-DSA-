"""
Dephaze DSA — Layer Boundary Test
==================================
Falsifiable prediction: peak R² at layer floor(L * phi^-2) + 1

Paper: Fossil Hallucination Decoding v1.4 (Dewer, 2026)
DOI:   10.5281/zenodo.20020443

Results:
  Mistral-7B:   L=32, predicted L13, actual L13  ✓ exact
  OPT-1.3B:     L=24, predicted L10, actual L9   ~ delta=1
  Pythia-6.9B:  L=32, predicted L13, actual L18  ✗ delta=5
  GPT-2 medium: L=24, predicted L10, actual L22  ✗ delta=12

Conclusion: phi^-2 boundary predicts peak in 2/4 architectures.
The high R² throughout (>0.63 in all models) is robust.
"""

# !pip install -q transformers accelerate scipy numpy

import numpy as np
from scipy.stats import pearsonr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

PHI  = (1 + 5**0.5) / 2
PHI3 = PHI**3

# ── Model selection ─────────────────────────────────────────
# Modify TEST_MODEL to test different architectures
TEST_MODEL = "facebook/opt-1.3b"   # L=24, d=2048
# TEST_MODEL = "gpt2-medium"        # L=24, d=1024
# TEST_MODEL = "EleutherAI/pythia-6.9b"  # L=32, d=4096
USE_4BIT   = False

print("LAYER BOUNDARY FALSIFIABLE PREDICTION")
print(f"Model: {TEST_MODEL}")
print()

# Predictions for all architectures
configs = [
    ("GPT-2 small",    12,  768), ("GPT-2 medium",  24, 1024),
    ("GPT-2 large",    36, 1280), ("GPT-2 xl",      48, 1600),
    ("Pythia-6.9B",    32, 4096), ("Mistral-7B",    32, 4096),
    ("OPT-1.3B",       24, 2048), ("OPT-6.7B",      32, 4096),
]
print(f"  {'Model':<16} {'L':>4}  {'phi^-2':>8}  {'Predicted peak':>15}")
print("  " + "-"*50)
for name, Lc, dc in configs:
    b = int(Lc * PHI**-2)
    print(f"  {name:<16} {Lc:>4}  L{b:>6}  L{b+1:>13}")
print()

# ── Load model ─────────────────────────────────────────────
if USE_4BIT:
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        TEST_MODEL, quantization_config=bnb,
        device_map="auto", output_hidden_states=True)
else:
    model = AutoModelForCausalLM.from_pretrained(
        TEST_MODEL, dtype=torch.float16,
        device_map="auto", output_hidden_states=True)

tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

L = model.config.num_hidden_layers
D = model.config.hidden_size
try:
    FF = model.config.intermediate_size
except AttributeError:
    FF = D * 4

boundary   = int(L * PHI**-2)
pred_peak  = boundary + 1

k6,k5,k4,k3,k2 = [round(D * PHI**-n) for n in [6,5,4,3,2]]
STEPS  = [k6,k5,k4,k3,k2]
K4_IDX = 2

print(f"d={D}, L={L}, ff/d={FF/D:.3f}")
print(f"phi^-2 boundary: L{boundary}")
print(f"PREDICTED PEAK:  L{pred_peak}")
print(f"k-lags: {STEPS}, Fibonacci: {k4+k3==k2}")

# ── QA pairs (same as crossmodel test) ─────────────────────
PAIRS = [
    ("What is the capital of France?",
     "The capital of France is Paris.",
     "The capital of France is London."),
    ("Who wrote Romeo and Juliet?",
     "Shakespeare wrote Romeo and Juliet.",
     "Marlowe wrote Romeo and Juliet."),
    ("What planet is closest to the Sun?",
     "Mercury is closest to the Sun.",
     "Venus is closest to the Sun."),
    ("How many sides does a triangle have?",
     "A triangle has three sides.",
     "A triangle has four sides."),
    ("What is the largest ocean?",
     "The Pacific is the largest ocean.",
     "The Atlantic is the largest ocean."),
    ("Who painted the Mona Lisa?",
     "Leonardo da Vinci painted the Mona Lisa.",
     "Michelangelo painted the Mona Lisa."),
    ("What gas do plants absorb?",
     "Plants absorb carbon dioxide.",
     "Plants absorb oxygen."),
    ("How many continents are there?",
     "There are seven continents.",
     "There are six continents."),
    ("What is the chemical symbol for gold?",
     "The chemical symbol for gold is Au.",
     "The chemical symbol for gold is Go."),
    ("Who was the first US president?",
     "George Washington was the first US president.",
     "Abraham Lincoln was the first US president."),
    ("What is the speed of light?",
     "The speed of light is approximately 300,000 km/s.",
     "The speed of light is approximately 150,000 km/s."),
    ("How many bones are in the human body?",
     "The human body has 206 bones.",
     "The human body has 186 bones."),
    ("What is the boiling point of water?",
     "Water boils at 100 degrees Celsius.",
     "Water boils at 90 degrees Celsius."),
    ("Who discovered penicillin?",
     "Alexander Fleming discovered penicillin.",
     "Louis Pasteur discovered penicillin."),
    ("What is the smallest planet?",
     "Mercury is the smallest planet.",
     "Pluto is the smallest planet."),
    ("How many hours in a day?",
     "There are 24 hours in a day.",
     "There are 12 hours in a day."),
    ("What language do Brazilians speak?",
     "Brazilians speak Portuguese.",
     "Brazilians speak Spanish."),
    ("Who invented the telephone?",
     "Alexander Graham Bell invented the telephone.",
     "Thomas Edison invented the telephone."),
    ("What is the longest river?",
     "The Nile is the longest river.",
     "The Amazon is the longest river."),
    ("How many players in a soccer team?",
     "A soccer team has eleven players.",
     "A soccer team has ten players."),
    ("What is the freezing point of water?",
     "Water freezes at 0 degrees Celsius.",
     "Water freezes at 32 degrees Celsius."),
    ("Who developed the theory of relativity?",
     "Einstein developed the theory of relativity.",
     "Newton developed the theory of relativity."),
    ("How many days in a leap year?",
     "A leap year has 366 days.",
     "A leap year has 365 days."),
    ("What is the currency of Japan?",
     "The currency of Japan is the yen.",
     "The currency of Japan is the yuan."),
    ("Who wrote the Odyssey?",
     "Homer wrote the Odyssey.",
     "Virgil wrote the Odyssey."),
    ("What is the largest planet?",
     "Jupiter is the largest planet.",
     "Saturn is the largest planet."),
    ("How many colors in a rainbow?",
     "A rainbow has seven colors.",
     "A rainbow has six colors."),
    ("What is the chemical formula for water?",
     "The chemical formula for water is H2O.",
     "The chemical formula for water is HO2."),
    ("Who invented the light bulb?",
     "Thomas Edison invented the light bulb.",
     "Nikola Tesla invented the light bulb."),
    ("What is the tallest mountain?",
     "Mount Everest is the tallest mountain.",
     "K2 is the tallest mountain."),
]
N = len(PAIRS)

# ── Hidden state extraction ─────────────────────────────────
def get_k_profile(text):
    inp = tokenizer(text, return_tensors="pt",
                    truncation=True, max_length=64,
                    padding=True).to(model.device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    profile = []
    for l in range(1, L+1):
        hs  = out.hidden_states[l][0].float().cpu().numpy()
        h_m = np.abs(hs).mean(axis=0)
        h_m = np.where(h_m < 1e-10, 1e-10, h_m)
        t   = np.log(h_m) / np.log(PHI3)
        c   = [float(pearsonr(t[:-k], t[k:])[0])
               if len(t) > max(STEPS) else 0.0
               for k in STEPS]
        profile.append(c)
    return np.array(profile)

T5 = np.zeros((N, L, 5))
H5 = np.zeros((N, L, 5))
for i, (q, ta, ha) in enumerate(PAIRS):
    if i % 5 == 0:
        print(f"  {i}/{N}")
    T5[i] = get_k_profile(q + " " + ta)
    H5[i] = get_k_profile(q + " " + ha)

# ── Fossil classification ───────────────────────────────────
delta      = T5 - H5
delta_mean = delta.mean(axis=0)
dc    = np.array([pearsonr(delta[i, min(boundary,L-1), :],
                           delta_mean[min(boundary,L-1), :])[0]
                  for i in range(N)])
F_idx = np.where(dc < 0)[0]
print(f"\nFossil: {len(F_idx)}/{N}")

# ── R² per layer ────────────────────────────────────────────
print(f"\nR² (k4={k4} scale) — all layers:")
print(f"PREDICTION: peak @ L{pred_peak}  (phi^-2 boundary = L{boundary})")
print()

r2_per_layer = []
for l in range(L):
    if len(F_idx) >= 3:
        r2 = float(pearsonr(-H5[F_idx, l, K4_IDX],
                             T5[F_idx, l, K4_IDX])[0]**2)
    else:
        r2 = np.nan
    r2_per_layer.append(r2)

r2_arr   = np.array(r2_per_layer)
peak_l   = int(np.nanargmax(r2_arr)) + 1
peak_r2  = float(np.nanmax(r2_arr))

for l, r2 in enumerate(r2_per_layer):
    bar          = "█" * int(r2 * 30) if not np.isnan(r2) else ""
    b_mark       = " ←φ⁻² boundary" if (l+1) == boundary   else ""
    pred_mark    = " ←PREDICTED"    if (l+1) == pred_peak   else ""
    actual_mark  = " ←ACTUAL PEAK"  if (l+1) == peak_l      else ""
    print(f"  L{l+1:>2}  {r2:.3f}  {bar}{b_mark}{pred_mark}{actual_mark}")

# ── Decision ───────────────────────────────────────────────
delta_l = abs(peak_l - pred_peak)
verdict = ("✓ EXACT"        if delta_l == 0 else
           f"~ CLOSE (Δ={delta_l})" if delta_l <= 2 else
           f"✗ MISS  (Δ={delta_l})")

print(f"\nFALSIFICATION DECISION")
print(f"  Predicted peak: L{pred_peak}")
print(f"  Actual peak:    L{peak_l}  (R²={peak_r2:.3f})")
print(f"  Verdict:        {verdict}")
print()

print("ACCUMULATED RESULTS:")
print(f"  {'Model':<14} {'L':>3}  {'Pred':>5}  {'Peak':>5}  {'R²':>6}  Verdict")
print("  " + "-"*52)
known = [
    ("Mistral-7B",    32, 13, 13, 0.971, "✓"),
    ("OPT-1.3B",      24, 10,  9, 0.991, "~Δ1"),
    ("Pythia-6.9B",   32, 13, 18, 0.983, "✗Δ5"),
    ("GPT-2 medium",  24, 10, 22, 0.972, "✗Δ12"),
]
for name, Lk, pred, actual, r2k, verd in known:
    print(f"  {name:<14} {Lk:>3}  L{pred:>3}   L{actual:>3}  {r2k:>6.3f}  {verd}")
tag = TEST_MODEL.split("/")[-1]
print(f"  {tag[:14]:<14} {L:>3}  L{pred_peak:>3}   L{peak_l:>3}  {peak_r2:>6.3f}  {verdict[:6]}")

"""
Dephaze DSA — ff/d Structural Test
====================================
Tests whether ff/d = 4 (exact match to phi^3 - phi^-3 = 4)
produces stronger Phi^3_ext signal than ff/d != 4.

Paper: Fossil Hallucination Decoding v1.4 (Dewer, 2026)
DOI:   10.5281/zenodo.20020443

Results at d=4096 (identical dimension, different ff/d):
  Mistral-7B    ff/d=3.500: T>H=31/32, effect=+0.003021
  Pythia-6.9B   ff/d=4.000: T>H=26/32, effect=+0.002074
  OpenLLaMA-7B  ff/d=2.688: T>H=26/32, effect=+0.003224

Max/min effect ratio = 1.55x  (threshold: 2x)

Conclusion: ff/d does NOT determine signal strength.
The dominant factor is d (hidden dimension), not ff/d.
"""

import subprocess, sys, importlib, os

def _pip(spec):
    subprocess.check_call([sys.executable, "-m", "pip",
                           "install", "-q", spec],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)

for pkg in ["transformers", "accelerate", "scipy"]:
    try:
        importlib.import_module(pkg)
    except ImportError:
        print(f"Installing {pkg}..."); _pip(pkg)

import numpy as np
from scipy.stats import pearsonr, binomtest, ttest_ind
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PHI  = (1 + 5**0.5) / 2
PHI3 = PHI**3

print(f"phi^3 - phi^-3 = {PHI3 - PHI**-3:.6f}  (exact = 4.000000)")
print()

# ── Model selection ──────────────────────────────────────────────
# ff/d = 4.0 EXACT: Pythia-6.9B, OPT-*, GPT-2-*
# ff/d = 3.5:       Mistral-7B   (SwiGLU)
# ff/d = 2.69:      OpenLLaMA-7B (SwiGLU)

TEST_MODEL = "openlm-research/open_llama_7b"  # ff/d=2.688, d=4096
# TEST_MODEL = "EleutherAI/pythia-6.9b"        # ff/d=4.000, d=4096
# TEST_MODEL = "mistralai/Mistral-7B-v0.3"     # ff/d=3.500, d=4096

print(f"Testing: {TEST_MODEL}")
print()

# ── Smart model loader ───────────────────────────────────────────
# Tries 4-bit first, falls back to fp16 if bitsandbytes unavailable

def load_model(model_name):
    """Load model: 4-bit if possible, fp16 otherwise."""

    # Try 4-bit
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa — just check it imports
        # Also verify transformers can use it
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        m = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            device_map="auto",
            output_hidden_states=True,
        )
        print("Loaded in 4-bit")
        return m
    except Exception as e:
        print(f"4-bit unavailable ({type(e).__name__}), trying fp16...")

    # Try fp16
    try:
        m = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
            output_hidden_states=True,
        )
        print("Loaded in fp16")
        return m
    except Exception as e:
        print(f"fp16 failed ({type(e).__name__}), trying fp32...")

    # Final fallback: fp32 (slow but always works)
    m = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        output_hidden_states=True,
    )
    print("Loaded in fp32")
    return m


model     = load_model(TEST_MODEL)
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

k6,k5,k4,k3,k2 = [round(D * PHI**-n) for n in [6,5,4,3,2]]
STEPS = [k6,k5,k4,k3,k2]

print(f"d={D}, L={L}, ff={FF}, ff/d={FF/D:.4f}")
print(f"delta from 4: {abs(FF/D-4.0):.4f}")
print(f"k-lags: {STEPS}, Fibonacci: {k4+k3==k2}")
print()

# ── QA pairs ────────────────────────────────────────────────────
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

# ── Hidden state + Phi^3_ext extraction ─────────────────────────
def get_profiles(text):
    """Returns k-profile [L,5] and Phi^3_ext [L]."""
    inp = tokenizer(text, return_tensors="pt",
                    truncation=True, max_length=64,
                    padding=True).to(model.device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    k_profile, phi3_ext = [], []
    for l in range(1, L+1):
        hs  = out.hidden_states[l][0].float().cpu().numpy()
        h_m = np.abs(hs).mean(axis=0)
        h_m = np.where(h_m < 1e-10, 1e-10, h_m)
        t   = np.log(h_m) / np.log(PHI3)
        c   = [float(pearsonr(t[:-k], t[k:])[0])
               if len(t) > max(STEPS) else 0.0
               for k in STEPS]
        ext = (c[1] + c[2]) - (c[3] + c[4])
        k_profile.append(c)
        phi3_ext.append(ext)
    return np.array(k_profile), np.array(phi3_ext)

T5    = np.zeros((N, L, 5))
H5    = np.zeros((N, L, 5))
T_ext = np.zeros((N, L))
H_ext = np.zeros((N, L))

for i, (q, ta, ha) in enumerate(PAIRS):
    if i % 5 == 0:
        print(f"  {i}/{N}")
    T5[i], T_ext[i] = get_profiles(q + " " + ta)
    H5[i], H_ext[i] = get_profiles(q + " " + ha)

tag = TEST_MODEL.split("/")[-1].replace("-", "_")
np.save(f"T5_{tag}.npy",   T5)
np.save(f"H5_{tag}.npy",   H5)
np.save(f"Text_{tag}.npy", T_ext)
np.save(f"Hext_{tag}.npy", H_ext)
print(f"Saved: T5{T5.shape}")

# ── Fossil classification ────────────────────────────────────────
delta      = T5 - H5
delta_mean = delta.mean(axis=0)
best_l     = min(28, L-1)
dc = np.array([pearsonr(delta[i, best_l, :],
               delta_mean[best_l, :])[0] for i in range(N)])
F_idx = np.where(dc < 0)[0]
print(f"\nFossil: {len(F_idx)}/{N}")

# ── Phi^3_ext T>H analysis ───────────────────────────────────────
t_gt_h, effects, pvals = [], [], []
for l in range(L):
    T_l = T_ext[:, l]; H_l = H_ext[:, l]
    t_gt_h.append(T_l.mean() > H_l.mean())
    effects.append(float(np.mean(T_l - H_l)))
    _, p = ttest_ind(T_l, H_l)
    pvals.append(p)

n_T_gt_H    = sum(t_gt_h)
mean_effect = float(np.mean(effects))
p_binom     = binomtest(n_T_gt_H, L, 0.5).pvalue

# ── Mirror correlation ───────────────────────────────────────────
all_r = []
for l in range(L):
    for j in range(5):
        if len(F_idx) >= 3:
            r, _ = pearsonr(H5[F_idx, l, j], -T5[F_idx, l, j])
            all_r.append(float(r))
mirror_mean = float(np.mean(all_r)) if all_r else float("nan")

# ── Final cross-table ────────────────────────────────────────────
print()
print("=" * 68)
print("CROSS-TABLE — ff/d STRUCTURAL TEST  (d=4096)")
print("=" * 68)
print()
print(f"  {'Model':<16} {'d':>5}  {'ff/d':>6}  {'T>H':>5}  {'Effect':>9}"
      f"  {'p_bin':>10}  {'Mirror':>7}")
print("  " + "-"*68)

known = [
    ("Mistral-7B",   4096, 3.500, 31, 32, +0.003021, 1.54e-8,  -0.849),
    ("Pythia-6.9B",  4096, 4.000, 26, 32, +0.002074, 5.35e-4,  -0.892),
    ("OpenLLaMA-7B", 4096, 2.688, 26, 32, +0.003224, 5.35e-4,  -0.861),
]
for name, d, ffd, tgt, Lk, eff, pb, mir in known:
    print(f"  {name:<16} {d:>5}  {ffd:>6.3f}  {tgt:>2}/{Lk}"
          f"  {eff:>+9.6f}  {pb:>10.2e}  {mir:>7.3f}")

if D == 4096:
    print(f"  {tag[:16]:<16} {D:>5}  {FF/D:>6.3f}"
          f"  {n_T_gt_H:>2}/{L}"
          f"  {mean_effect:>+9.6f}  {p_binom:>10.2e}"
          f"  {mirror_mean:>7.3f}  <- NOW")

# ── Decision ─────────────────────────────────────────────────────
effects_all = [0.003021, 0.002074, 0.003224, mean_effect]
ratio = max(effects_all) / max(min(effects_all), 1e-9)

print()
print("DECISION:")
print(f"  Max/min effect ratio: {ratio:.2f}x  (threshold: 2x)")
if ratio < 2.0:
    print("  ff/d DOES NOT MATTER — d is the dominant factor")
    print("  phi^3-phi^-3=4 is algebraically exact, not causally dominant")
else:
    print("  ff/d MATTERS — significant difference across ratios")

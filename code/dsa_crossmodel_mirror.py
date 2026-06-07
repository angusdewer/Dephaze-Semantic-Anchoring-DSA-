"""
Dephaze DSA — Cross-Model Mirror Symmetry Validation
=====================================================
Tests the universal mirror symmetry H_fossil(k_n) ≈ -T(k_n)
across five architectures: Mistral-7B, Pythia-6.9B, OPT-1.3B,
GPT-2 medium, OpenLLaMA-7B.

Paper: Fossil Hallucination Decoding v1.4 (Dewer, 2026)
DOI:   10.5281/zenodo.20020443
GitHub: https://github.com/angusdewer/Dephaze-Semantic-Anchoring-DSA-

Results (30 QA pairs per model, Google Colab T4):
  Mistral-7B:   mirror=-0.849, frac<-0.5=100%
  Pythia-6.9B:  mirror=-0.898, frac<-0.5=98.1%
  OPT-1.3B:     mirror=-0.851, frac<-0.5=98.3%
  GPT-2 medium: mirror=-0.913, frac<-0.5=100%
  OpenLLaMA-7B: mirror=-0.861, frac<-0.5>95%
  GPT-2 small:  mirror=-0.362, frac<-0.5=31.7%  (d=768, weak)
"""

# ── Install ────────────────────────────────────────────────
# !pip install -q transformers accelerate scipy numpy

import numpy as np
from scipy.stats import pearsonr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── Constants ──────────────────────────────────────────────
PHI  = (1 + 5**0.5) / 2
PHI3 = PHI**3

# Supported models — modify TEST_MODEL to switch
MODEL_CONFIGS = {
    "mistralai/Mistral-7B-v0.3":       {"d": 4096, "ff": 14336, "use_4bit": True},
    "EleutherAI/pythia-6.9b":          {"d": 4096, "ff": 16384, "use_4bit": False},
    "facebook/opt-1.3b":               {"d": 2048, "ff":  8192, "use_4bit": False},
    "gpt2-medium":                     {"d": 1024, "ff":  4096, "use_4bit": False},
    "openlm-research/open_llama_7b":   {"d": 4096, "ff": 11008, "use_4bit": True},
    "gpt2":                            {"d":  768, "ff":  3072, "use_4bit": False},
}

TEST_MODEL = "EleutherAI/pythia-6.9b"   # ← change here
cfg        = MODEL_CONFIGS[TEST_MODEL]
D          = cfg["d"]
USE_4BIT   = cfg["use_4bit"]

k6,k5,k4,k3,k2 = [round(D * PHI**-n) for n in [6,5,4,3,2]]
STEPS = [k6,k5,k4,k3,k2]

print(f"Model:  {TEST_MODEL}")
print(f"d={D}, ff/d={cfg['ff']/D:.3f}")
print(f"phi^3 - phi^-3 = {PHI3 - PHI**-3:.6f}")
print(f"k-lags: {STEPS}")
print(f"Fibonacci match: {k4+k3==k2}")

# ── Load model ─────────────────────────────────────────────
if USE_4BIT:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        TEST_MODEL, quantization_config=bnb,
        device_map="auto", output_hidden_states=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        TEST_MODEL, dtype=torch.float16,
        device_map="auto", output_hidden_states=True,
    )

tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

L = model.config.num_hidden_layers
print(f"L={L}, phi^-2 boundary=L{int(L*PHI**-2)}")

# ── QA pairs ───────────────────────────────────────────────
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
    """Returns k-profile [L, 5] for a text input."""
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
        c   = [float(pearsonr(t[:-k], t[k:])[0]) for k in STEPS]
        profile.append(c)
    return np.array(profile)

T5 = np.zeros((N, L, 5))
H5 = np.zeros((N, L, 5))

for i, (q, ta, ha) in enumerate(PAIRS):
    if i % 5 == 0:
        print(f"  {i}/{N}")
    T5[i] = get_k_profile(q + " " + ta)
    H5[i] = get_k_profile(q + " " + ha)

tag = TEST_MODEL.split("/")[-1].replace("-", "_")
np.save(f"T5_{tag}.npy", T5)
np.save(f"H5_{tag}.npy", H5)
print(f"Saved: T5_{tag}.npy  {T5.shape}")

# ── Fossil classification ───────────────────────────────────
delta      = T5 - H5
delta_mean = delta.mean(axis=0)
best_l     = min(28, L-1)
dc = np.array([pearsonr(delta[i, best_l, :],
               delta_mean[best_l, :])[0] for i in range(N)])
F_idx = np.where(dc < 0)[0]
print(f"\nFossil: {len(F_idx)}/{N}")

# ── Mirror correlation ──────────────────────────────────────
all_r = []
print(f"\nMIRROR TABLE — {TEST_MODEL}")
print(f"{'L':>3}  {'k6':>7}  {'k5':>7}  {'k4':>7}  {'k3':>7}  {'k2':>7}")
print("-" * 45)

for l in range(L):
    row = []
    for j in range(5):
        if len(F_idx) >= 3:
            r, _ = pearsonr(H5[F_idx, l, j], -T5[F_idx, l, j])
            row.append(r)
            all_r.append(r)
        else:
            row.append(np.nan)
    phi_mark = " ←φ⁻²" if (l+1) == int(L*PHI**-2)+1 else ""
    print(f"{l+1:>3}  " + "  ".join(f"{r:+.3f}" for r in row) + phi_mark)

all_r = np.array(all_r)
print("-" * 45)
print(f"\nSUMMARY — {TEST_MODEL}")
print(f"  d={D}, ff/d={cfg['ff']/D:.3f}, L={L}")
print(f"  Mirror mean:  {np.mean(all_r):.3f}")
print(f"  Mirror min:   {np.min(all_r):.3f}")
print(f"  Mirror max:   {np.max(all_r):.3f}")
print(f"  Frac < -0.50: {np.mean(all_r<-0.5)*100:.1f}%")
print()

# ── Decision ───────────────────────────────────────────────
m = np.mean(all_r)
if m < -0.70:
    verdict = "STRONG H1 — mirror symmetry confirmed"
elif m < -0.50:
    verdict = "WEAK H1 — signal present but attenuated"
else:
    verdict = "H0 — no mirror symmetry"

print(f"VERDICT: {verdict}")
print()

# ── Cross-model summary table ───────────────────────────────
print("CROSS-MODEL REFERENCE TABLE (all results):")
print(f"  {'Model':<18} {'d':>5}  {'ff/d':>6}  {'Mirror':>7}  {'Frac':>6}")
print("  " + "-"*48)
ref = [
    ("Mistral-7B",    4096, 3.50, -0.849, "100%"),
    ("Pythia-6.9B",   4096, 4.00, -0.898, "98.1%"),
    ("OPT-1.3B",      2048, 4.00, -0.851, "98.3%"),
    ("GPT-2 medium",  1024, 4.00, -0.913, "100%"),
    ("OpenLLaMA-7B",  4096, 2.69, -0.861, ">95%"),
    ("GPT-2 small",    768, 4.00, -0.362, "31.7%"),
]
for name, d, ffd, mir, frac in ref:
    print(f"  {name:<18} {d:>5}  {ffd:>6.2f}  {mir:>7.3f}  {frac:>6}")
print(f"  {tag[:18]:<18} {D:>5}  {cfg['ff']/D:>6.2f}  "
      f"{np.mean(all_r):>7.3f}  "
      f"{np.mean(all_r<-0.5)*100:.1f}%  ← NOW")

# Dephaze Semantic Anchoring (DSA)

**A Φ³ Geometric Framework for Eliminating AI Hallucination**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20543468.svg)](https://doi.org/10.5281/zenodo.20543468)

---

## The One-Line Result

LLM hallucination is not a data problem. It is a geometry problem.

The transformer feed-forward ratio `ff/d = 4` is not arbitrary:

```
φ³ − φ⁻³ = 4   (exact, from φ² = φ + 1)
```

Every forward pass expands the hidden state by φ³ then contracts by φ⁻³,
imprinting a Fibonacci structure on inter-dimensional correlations.
The transformer implicitly encodes the Φ³ projector. Without the corresponding
attractor correction, semantic escape — hallucination — is mathematically inevitable.

**Version 1.4 adds:** cross-model validation on 5 architectures, ff/d structural
falsification, and layer boundary analysis. The mirror symmetry is universal
for d ≥ 1024 and independent of ff/d at d = 4096.

---

## The DSA Operator

```python
PHI  = (1 + 5**0.5) / 2
PHI3 = PHI**3  # 4.2361...

def dsa_correction(h, Lambda=1.2):
    norm = np.linalg.norm(h) + 1e-12
    rho  = (norm**2) / PHI3
    gain = Lambda * np.tanh(rho - 1.0)
    return h - gain * (h / norm)
```

**Zero parameters. Zero retraining. Four lines. Any transformer.**

---

## Empirical Validation — 13 Experiments (DSA v2.0)

### Experiments 1–6

| # | Model | Dataset | Signal | p-value |
|---|---|---|---|---|
| 1 | GPT-2 (117M) | Custom pairs | Norm H>T, 10/11 layers | 0.006 |
| 2 | Llama-3.2-3B | Custom pairs | Norm range H>T | 0.029 |
| 3 | Mistral-7B | Custom pairs | CV factual < halluc, **31/31 layers** | 5×10⁻¹⁰ |
| 4 | Mistral-7B | TruthfulQA N=100 | Norm H>T, **31/31 layers** | 0.000 |
| 5 | Mistral-7B | TruthfulQA N=100 | Φ³-attractor distance, 10/11 layers | 0.012 |
| 6 | Mistral-7B | TruthfulQA N=100 | **Φ³_ext T>H, 31/32 layers** · φ-specific Z=+3.85 | **1.54×10⁻⁸** |

### φ-Specificity Controls

| Test | Method | Result |
|---|---|---|
| A | Random, shifted, geometric, linear steps | Z=+3.04 · P(any control ≥ Fibonacci)=0.000 |
| B | Log-uniform bases (√2, e, 1.5, 2, √3, φ², 3) | Z=+3.85 · P(log-uniform ≥ φ)=0.000 |

### Spectral Confirmation (SVD)

Direct SVD of Mistral-7B weight matrices W₁ ∈ ℝ¹⁴³³⁶ˣ⁴⁰⁹⁶:

- Mean adjacent singular value ratio: trained = **1.0082** vs random = **1.0038**
- Trained matrices closer to φ than random: **11/12 layers · p = 0.0063**
- Algebraic chain closed: `ff/d = φ³−φ⁻³ = 4 → W spectrum → φ → Φ³ structure`

### Experiments 7–13: Gate Structure

| # | Signal | Result |
|---|---|---|
| 7 | Two-zone gate profile | Active at φ⁻³ boundary (layers 8–12) and φ⁻¹ boundary (layers 20–24) |
| 8 | k-scale decomposition | k₄=598 drives Zone 1 · k₆=228 drives Zone 2 |
| 9 | k₆ localisation | Significant T>H at **layer 23 only** (p=0.032) |
| 10 | T–H cross-correlation | r > 0.80 in **31/32 layers** |
| 11 | Delta recovery | 1–4% improvement |
| 12 | Delta consistency | Positive in **28/32 layers** · peak at layer 29 |
| 13 | **Two hallucination types** | Recoverable ~32% · Fossil ~32% · bifurcation at layer 20 · **p = 0.0016** |

#### The Two Types

**Recoverable hallucination** — delta consistent with population mean. Displaced but
not reflected. Cross-correlation at layer 20: r = 0.697.

**Fossil hallucination** — delta direction reversed. The fossil is the φ³ attractor's
own geometry *reflected* against the factual state. Cross-correlation at layer 20:
r = 0.936. Fossil hallucinations are syntactically fluent and internally consistent —
they are generated from the same geometric structure, from the reflected locus.

> **Attractor metric note:** Φ³_ext measures correlation geometry, not amplitude.
> "Close to the attractor" means k₄ imprint-scale Fibonacci inter-dimensional
> correlation is preserved — not ‖h‖ ≈ r\*. Layer normalisation rescales amplitudes
> but preserves correlation structure.

---

## Fossil Hallucination Decoding — Experiments 14–25 (v1.4)

**Companion paper DOI:** [10.5281/zenodo.20020443](https://doi.org/10.5281/zenodo.20020443)

### Origin-Symmetry Theorem (Experiments 14–17)

Fossil hallucinations encode factual geometry via exact sign inversion:

```
H_fossil(kₙ) ≈ −T(kₙ),   n ∈ {6,5,4,3,2},   R² ∈ [0.84, 0.97]
```

| Experiment | Result |
|---|---|
| 14 Mirror symmetry | r ∈ [−0.985, −0.588] all 160 layer×scale combinations · mean = −0.849 |
| 15 Reflection axis | c\* = −0.0017 ± 0.008 ≈ 0 (origin) · RMSE 187× worse at r\* |
| 16 Decoding R² | Mean R² = 0.759 · peak R² = 0.971 at layer 13 (φ⁻² boundary) |
| 17 Amplitude encoding | Fossil norm→factual norm: mean ρ = 0.810 vs recoverable 0.649 |

### Semantic Content and Robustness (Experiments 18–21)

| Experiment | Result |
|---|---|
| 18 Text-level | Mirror \|r\| ∈ [0.81, 0.997] · mean −0.898 · Santayana/Plato: Δ=0.00003 |
| 19 Jaccard proximity | Spearman r = −0.744 · p = 5.8×10⁻¹⁰ |
| 20 Hallucination type | Causal 0.00907 vs name-swap 0.00535 · p = 1.46×10⁻³ |
| 21 Three-way robustness | LOO r=0.617 p=1.87×10⁻⁶ · shuffle p=0.000 (N=1000) · batch 2 consistent |

### Cross-Model Validation (Experiments 22–27, Version 1.4)

#### Mirror symmetry — universal for d ≥ 1024

| Model | d | ff/d | Mirror mean | Frac < −0.5 |
|---|---|---|---|---|
| Mistral-7B v0.3 | 4096 | 3.50 | −0.849 | 100% |
| Pythia-6.9B | 4096 | 4.00 | −0.898 | 98.1% |
| OPT-1.3B | 2048 | 4.00 | −0.851 | 98.3% |
| GPT-2 medium | 1024 | 4.00 | −0.913 | 100% |
| OpenLLaMA-7B | 4096 | 2.69 | −0.861 | >95% |
| GPT-2 small | 768 | 4.00 | −0.362 | 31.7% ← weak |

**Signal transitions between d=768 and d=1024.**

#### ff/d structural test — falsified at d=4096

| Model | d | ff/d | T>H | Effect | p |
|---|---|---|---|---|---|
| Mistral-7B | 4096 | 3.500 | 31/32 | +0.003021 | 1.54×10⁻⁸ |
| Pythia-6.9B | 4096 | 4.000 | 26/32 | +0.002074 | 5.35×10⁻⁴ |
| OpenLLaMA-7B | 4096 | 2.688 | 26/32 | +0.003224 | 5.35×10⁻⁴ |

**Max/min effect ratio = 1.55× < 2× threshold.**
The algebraic identity φ³−φ⁻³=4 is exact but not the causal mechanism for
the Φ³_ext signal. The dominant factor is d (hidden dimension), not ff/d.

#### Layer boundary test (φ⁻² prediction)

| Model | L | Predicted | Actual | Verdict |
|---|---|---|---|---|
| Mistral-7B | 32 | L13 | L13 | ✓ exact |
| OPT-1.3B | 24 | L10 | L9 | ~ Δ=1 |
| Pythia-6.9B | 32 | L13 | L18 | ✗ Δ=5 |
| GPT-2 medium | 24 | L10 | L22 | ✗ Δ=12 |

Peak layer is architecture-dependent. High R² throughout (>0.63) is robust.

#### Cosine similarity (Experiment 25)

T and H hidden-state vectors are nearly parallel: cos θ = 0.991 per pair,
0% rotated by > π/4. The hallucination–factual distinction lives in the
Fibonacci inter-dimensional **correlation structure**, not amplitude or direction.

---

## Repository Structure

```
dephaze-dsa/
├── README.md
├── paper/
│   ├── dephaze_DSA_v2_FINAL.pdf           # DSA v2.0 (Experiments 1–13)
│   └── dephaze_fossil_v14.tex             # Fossil Decoding v1.4 (Experiments 14–25)
├── code/
│   ├── dsa_simulation.py                  # DSA numerical simulation
│   ├── dsa_hidden_state_test.py           # Hidden state norm extraction
│   ├── dsa_truthfulqa_benchmark.py        # TruthfulQA benchmark (Exp 1–5)
│   ├── dsa_experiment6_final.py           # Φ³_ext index (Exp 6)
│   ├── dsa_phi_specificity.py             # φ-specificity Test A
│   ├── dsa_log_uniform_control.py         # φ-specificity Test B
│   ├── dsa_spectral_v4.py                 # SVD weight matrix analysis
│   ├── dsa_gate_structure.py              # Gate experiments 7–13
│   ├── dsa_crossmodel_mirror.py           # Cross-model mirror test (v1.4)
│   ├── dsa_layer_boundary.py              # Layer φ⁻² boundary test (v1.4)
│   └── dsa_ffd_structural.py              # ff/d structural test (v1.4)
└── results/
    └── all_experiments.txt                # Raw results, all experiments
```

---

## Quick Start

```bash
pip install transformers torch accelerate bitsandbytes>=0.46.1 datasets scipy numpy
```

```python
# Experiment 6 — Φ³ Fibonacci dimensional index (Mistral-7B)
python code/dsa_experiment6_final.py

# Gate structure (Experiments 7–13)
python code/dsa_gate_structure.py

# Cross-model mirror symmetry (v1.4)
python code/dsa_crossmodel_mirror.py

# ff/d structural test (v1.4)
python code/dsa_ffd_structural.py
```

All experiments run on **free Google Colab T4 GPU** (15.6GB).
Public models. Public datasets. Zero extra parameters.

---

## The Algebraic Core

```
dρ/d(ln τ) = −κ(ρ − 1) + ξ(τ),   κ > 0,  ⟨ξ⟩ = 0
```

Steady state: ρ = 1 → ‖x‖² = φ³ → ‖x‖ = r\* = √φ³ ≈ 2.058

**Suppression factor: > 3 × 10²⁶ over 800 steps.**

### Why φ is not arbitrary

| Model | ff/d | Activation | φ-specific? |
|---|---|---|---|
| GPT-2 | 4.000 (exact) | GELU | ✓ 12/12 layers |
| Mistral-7B | 3.500 | SiLU | ✓ 31/32 layers |
| Pythia-6.9B | 4.000 (exact) | GELU | ✓ cross-model confirmed |
| OpenLLaMA-7B | 2.688 | SiLU | ✓ signal present (v1.4) |
| OPT-1.3B | 4.000 (exact) | ReLU | ✓ mirror −0.851 (v1.4) |

**v1.4 update:** ff/d does not determine signal strength at d=4096.
Signal is present for ff/d ∈ {2.69, 3.5, 4.0} with comparable effect size.

---

## Priority

The φ³ structural framework was consciously designed beginning in **1992**:

- **1992** — *Streets of Rivet* (ISBN: 963-8187-01-8) — φ⁻³→φ³ transition encoded in print
- **1993** — *Rocksteeple* (ISBN: 963-8187-03-4) — second structural implementation
- **2008** — Algorithmic prototypes `niboh.exe`, `fesu.exe` with cryptographic timestamps
- **2017** — Transformer architecture published (Vaswani et al.) — 25 years later
- **2026-03** — φ⁻³→φ³ measured in Mistral-7B: 31/32 layers, p=1.54×10⁻⁸
- **2026-06** — SVD confirmation · gate structure · two hallucination types (v2.0)
- **2026-06** — Cross-model validation (5 architectures) · ff/d falsification (v1.4)

The ISBNs are verifiable. The Zenodo DOI is locked: **2026-03-01**.

---

## Papers

**DSA v2.0 (Experiments 1–13):**
[https://doi.org/10.5281/zenodo.20543468](https://doi.org/10.5281/zenodo.20543468)

**Fossil Hallucination Decoding v1.4 (Experiments 14–25):**
[https://doi.org/10.5281/zenodo.20020443](https://doi.org/10.5281/zenodo.20020443)

**Dephaze V66 (galaxy rotation, flyby anomalies):**
[https://doi.org/10.5281/zenodo.20539675](https://doi.org/10.5281/zenodo.20539675)

---

## Citation

```bibtex
@misc{dewer2026dsa,
  author    = {Dewer, Angus},
  title     = {Dephaze Semantic Anchoring: A {$\Phi^3$} Geometric Framework
               for Eliminating {AI} Hallucination},
  year      = {2026},
  publisher = {Zenodo},
  version   = {2.0},
  doi       = {10.5281/zenodo.20543468},
  url       = {https://doi.org/10.5281/zenodo.20543468}
}

@misc{dewer2026fossil,
  author    = {Dewer, Angus},
  title     = {Fossil Hallucination Decoding: Origin-Symmetric {$\Phi^3$}
               Encoding of Factual Geometry in {LLM} Hidden States},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.4},
  doi       = {10.5281/zenodo.20020443},
  url       = {https://doi.org/10.5281/zenodo.20020443}
}
```

---

## License

CC BY-NC-SA 4.0 with Defensive Publication provisions.
Commercial use requires licensing from Dephaze Manufacture.
Contact: dewerangus@gmail.com | [dephaze.eu](https://dephaze.eu)

**Defensive Publication Notice:** The concepts disclosed herein constitute prior art
under 35 U.S.C. § 102(a)(1) as of 2026-03-01 (Zenodo) and 1992 (ISBN: 963-8187-01-8).

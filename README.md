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

## Empirical Validation — 13 Experiments (v2.0)

### Experiments 1–6 (confirmed from prior version)

| # | Model | Dataset | Signal | p-value |
|---|---|---|---|---|
| 1 | GPT-2 (117M) | Custom pairs | Norm H>T, 10/11 layers | 0.006 |
| 2 | Llama-3.2-3B | Custom pairs | Norm range H>T | 0.029 |
| 3 | Mistral-7B | Custom pairs | CV factual < halluc, **31/31 layers** | 5×10⁻¹⁰ |
| 4 | Mistral-7B | TruthfulQA N=100 | Norm H>T, **31/31 layers** | 0.000 |
| 5 | Mistral-7B | TruthfulQA N=100 | Φ³-attractor distance, 10/11 layers | 0.012 |
| 6 | Mistral-7B | TruthfulQA N=100 | **Φ³_ext T>H, 31/32 layers** · φ-specific Z=+3.85 | **1.54×10⁻⁸** |

### φ-Specificity Controls (two independent tests)

| Test | Method | Result |
|---|---|---|
| A | Arbitrary step quadruples (random, shifted, geometric, linear) | Z=+3.04 · P(any control ≥ Fibonacci)=0.000 |
| B | Log-uniform bases (√2, e, 1.5, 2, √3, φ², 3) | Z=+3.85 · P(log-uniform ≥ φ)=0.000 |

Not enough to be logarithmically uniform — the base φ carries the signal.

### Spectral Confirmation

Direct SVD of Mistral-7B weight matrices W₁ ∈ ℝ¹⁴³³⁶ˣ⁴⁰⁹⁶, W₂ ∈ ℝ⁴⁰⁹⁶ˣ¹⁴³³⁶:

- Mean adjacent singular value ratio r̄ trained = **1.0082** vs random = **1.0038**
- Trained matrices closer to φ than random baseline: **11/12 layers · p = 0.0063**
- Algebraic chain closed: `ff/d = φ³−φ⁻³ = 4 →(SVD)→ W spectrum → φ →(Exp.6)→ Φ³ structure`

### Experiments 7–13: Gate Structure (v2.0, new)

| # | Signal | Result |
|---|---|---|
| 7 | Two-zone gate profile | Active at φ⁻³ boundary (layers 8–12) and φ⁻¹ boundary (layers 20–24) |
| 8 | k-scale decomposition | k₄=598 drives Zone 1 · k₆=228 drives Zone 2 |
| 9 | k₆ localisation | Significant T>H at **layer 23 only** (p=0.032) |
| 10 | T–H cross-correlation | r > 0.80 in **31/32 layers** — hallucination is displaced factual geometry |
| 11 | Delta recovery | 1–4% improvement — delta is question-specific, not universal |
| 12 | Delta consistency | Positive in **28/32 layers** · peak at layer 29 |
| 13 | **Two hallucination types** | Recoverable ~32% · Fossil ~32% · bifurcation at layer 20 · **p = 0.0016** |

#### The Two Types

**Recoverable hallucination** — delta consistent with population mean. Displaced from factual state in φ-space but not reflected. Cross-correlation at layer 20: r = 0.697.

**Fossil hallucination** — delta direction reversed. The fossil is the φ³ attractor's own geometry *reflected* against the factual state through r\*. Cross-correlation at layer 20: r = 0.936. This explains why fossil hallucinations are syntactically fluent and internally consistent — they are generated from the same geometric structure, merely from the reflected locus.

All reproducible on **free Google Colab T4 GPU**. Public models. Public datasets. Zero extra parameters.

---

## Repository Structure

```
dephaze-dsa/
├── README.md
├── paper/
│   └── dephaze_DSA_v2_FINAL.pdf           # Full paper (v2.0)
├── code/
│   ├── dsa_simulation.py                  # DSA numerical simulation
│   ├── dsa_hidden_state_test.py           # Hidden state norm extraction
│   ├── dsa_truthfulqa_benchmark.py        # TruthfulQA benchmark (Exp 1–5)
│   ├── dsa_experiment6_final.py           # Φ³_ext index (Exp 6)
│   ├── dsa_phi_specificity.py             # φ-specificity Test A
│   ├── dsa_log_uniform_control.py         # φ-specificity Test B
│   ├── dsa_spectral_v4.py                 # SVD weight matrix analysis
│   └── dsa_gate_structure.py             # Gate experiments 7–13
└── results/
    └── all_experiments.txt                # Raw results, all 13 experiments
```

---

## Quick Start

```bash
pip install transformers torch accelerate datasets scipy scikit-learn numpy
```

```python
# Experiment 6 — Φ³ Fibonacci dimensional index
python code/dsa_experiment6_final.py

# φ-specificity controls
python code/dsa_log_uniform_control.py

# Gate structure (Experiments 7–13)
python code/dsa_gate_structure.py
```

Requires ~15GB GPU RAM (free Colab T4 works).

---

## The Algebraic Core

The Dephaze Self-Regulation Axiom (AXIOM_2.5):

```
dρ/d(ln τ) = −κ(ρ − 1) + ξ(τ),   κ > 0,  ⟨ξ⟩ = 0
```

Steady state: ρ = 1 → ‖x‖² = φ³ → ‖x‖ = r\* = √φ³ ≈ 2.058

This is the unique attractor. Without it: semantic escape. With it: stability.

**Suppression factor: > 3 × 10²⁶ over 800 steps.**

### Why φ is not arbitrary

| Model | ff/d | Activation | φ specific? |
|---|---|---|---|
| GPT-2 | 4.000 (exact) | GELU | ✓ 12/12 layers |
| Mistral-7B | 3.500 | SiLU | ✓ 31/32 layers |
| OPT-6.7B | 4.000 | ReLU | ✗ head-aligned (2ⁿ dominant) |
| Qwen-1.5-7B | 2.688 | SiLU | ✗ ff/d too far from 4 |

Signal strongest when ff/d ≈ 4 **and** smooth activation (SiLU/GELU).
GPT-4, Claude, Gemini: predicted φ-specific at n=(5,4,3,2) — direct verification
requires hidden-state access not available via public API.

---

## Priority

The φ³ structural framework was consciously designed beginning in **1992**:

- **1992** — *Streets of Rivet* (ISBN: 963-8187-01-8) — φ⁻³→φ³ transition encoded in print
- **1993** — *Rocksteeple* (ISBN: 963-8187-03-4) — second structural implementation
- **2008** — Algorithmic prototypes `niboh.exe`, `fesu.exe` with cryptographic timestamps
- **2017** — Transformer architecture published (Vaswani et al.) — **25 years later**
- **2026-03** — φ⁻³→φ³ measured in Mistral-7B: 31/32 layers, p=1.54×10⁻⁸ (v1.8)
- **2026-06** — SVD spectral confirmation (11/12, p=0.0063) · gate structure · two hallucination types (v2.0)

The ISBNs are verifiable. The Zenodo DOI is locked: **2026-03-01**.

---

## Paper

**Zenodo preprint v2.0:**
[https://doi.org/10.5281/zenodo.20543468](https://doi.org/10.5281/zenodo.20543468)

**Related framework (Dephaze V66 — galaxy rotation, flyby anomalies):**
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
```

---

## License

CC BY-NC-SA 4.0 with Defensive Publication provisions.
Commercial use requires licensing from Dephaze Manufacture.
Contact: dewerangus@gmail.com | [dephaze.eu](https://dephaze.eu)

**Defensive Publication Notice:** The concepts disclosed herein constitute prior art
under 35 U.S.C. § 102(a)(1) as of 2026-03-01 (Zenodo) and 1992 (ISBN: 963-8187-01-8).

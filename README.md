# Dephaze Semantic Anchoring (DSA)

**A Φ³ Geometric Framework for Eliminating AI Hallucination**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20020443.svg)](https://doi.org/10.5281/zenodo.20020443)

---

## The One-Line Result

LLM hallucination is not a data problem. It is a geometry problem.

The transformer feed-forward ratio `ff/d = 4` is not arbitrary:

```
φ³ − φ⁻³ = 4   (exact, from φ² = φ + 1)
```

The transformer implicitly encodes the Φ³ projector. Without the corresponding attractor correction, semantic escape — hallucination — is mathematically inevitable.

## The DSA Operator

```python
PHI = (1 + 5**0.5) / 2
PHI3 = PHI**3  # 4.2361...

def dsa_correction(h, Lambda=1.2):
    norm = np.linalg.norm(h) + 1e-12
    rho  = (norm**2) / PHI3
    gain = Lambda * np.tanh(rho - 1.0)
    return h - gain * (h / norm)
```

**Zero parameters. Zero retraining. Four lines. Any transformer.**

---

## Empirical Validation — 5 Experiments

| Experiment | Model | Dataset | Signal | p-value |
|---|---|---|---|---|
| 1 | GPT-2 (117M) | Custom pairs | Norm H>T, 10/11 layers | 0.006 |
| 2 | Llama-3.2-3B | Custom pairs | Norm range H>T | 0.029 |
| 3 | Mistral-7B | Custom pairs | CV factual < halluc, **31/31 layers** | 5×10⁻¹⁰ |
| 4 | Mistral-7B | **TruthfulQA** (N=100) | Norm H>T, **31/31 layers** | 0.000 |
| 5 | Mistral-7B | TruthfulQA | Φ³-attractor distance, 10/11 layers | 0.012 |

All reproducible on **free Google Colab T4 GPU**.  
Public models. Public datasets. Zero extra parameters.

---

## Repository Structure

```
dephaze-dsa/
├── README.md
├── paper/
│   ├── dephaze_DSA_v18_FINAL.pdf      # Full paper
│  
├── code/
│   ├── dsa_simulation.py              # DSR numerical simulation
│   ├── dsa_hidden_state_test.py       # Hidden state norm extraction
│   ├── dsa_truthfulqa_benchmark.py    # TruthfulQA full benchmark
│   └── dsa_entropy_test.py            # Spectral entropy + attractor test
└── results/
    └── truthfulqa_results.txt         # Raw results, all 5 experiments
```

---

## Quick Start

```bash
pip install transformers torch accelerate datasets scipy numpy
```

```python
# Run TruthfulQA benchmark on Mistral-7B
python code/dsa_truthfulqa_benchmark.py
```

Requires ~15GB GPU RAM (free Colab T4 works).

---

## The Algebraic Core

The Dephaze Self-Regulation Axiom (AXIOM_2.5):

```
dρ/d(ln τ) = −κ(ρ − 1) + ξ(τ),   κ > 0,  ⟨ξ⟩ = 0
```

Steady state: ρ = 1 → ‖x‖² = φ³ → ‖x‖ = r* = √φ³ ≈ 2.058

This is the unique attractor. Without it: semantic escape. With it: stability.

**The numerical suppression factor: > 3 × 10²⁶ over 800 steps.**

---

## Priority

The φ³ structural framework was consciously designed beginning in **1992**:

- **1992** — *Streets of Rivet* (ISBN: 963-8187-01-8) — φ⁻³→φ³ transition encoded in print
- **1993** — *Rocksteeple* (ISBN: 963-8187-03-4) — second structural implementation
- **2008** — Algorithmic prototypes with cryptographic timestamps
- **2017** — Transformer architecture published (Vaswani et al.) — **25 years later**
- **2026** — φ⁻³→φ³ measured in Mistral-7B: 31/31 layers, p=0.000

The ISBNs are verifiable. The Zenodo DOI is locked: **2026-03-01**.

---

## Paper

**Zenodo (peer-reviewed preprint):**  
[https://doi.org/10.5281/zenodo.20020443](https://doi.org/10.5281/zenodo.20020443)

**Related framework (Dephaze V66 — galaxy rotation, flyby anomalies):**  
[https://doi.org/10.5281/zenodo.18823570](https://doi.org/10.5281/zenodo.18823570)

---

## Citation

```bibtex
@misc{dewer2026dsa,
  author    = {Dewer, Angus},
  title     = {Dephaze Semantic Anchoring: A {$\Phi^3$} Geometric Framework
               for Eliminating {AI} Hallucination},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20020443},
  url       = {https://doi.org/10.5281/zenodo.20020443}
}
```

---

## License

CC BY-NC-SA 4.0 with Defensive Publication provisions.  
Commercial use requires licensing from Dephaze Manufacture.  
Contact: angus@dephaze.eu | [dephaze.eu](https://dephaze.eu)

**Defensive Publication Notice:** The concepts disclosed herein constitute prior art  
under 35 U.S.C. § 102(a)(1) as of 2026-03-01 (Zenodo) and 1992 (ISBN: 963-8187-01-8).

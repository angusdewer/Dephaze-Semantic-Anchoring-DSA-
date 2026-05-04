"""
Dephaze Semantic Anchoring (DSA) — Hidden State Norm Extraction
===============================================================
DOI: 10.5281/zenodo.20020443

Tests DSA prediction on real LLM hidden states:
  - Factual prompts: more stable, closer to Omega_0 groundstate
  - Hallucinatory prompts: larger norm, less stable, Phi3-attractor

Requires: transformers, torch, accelerate, scipy, numpy
Hardware: Google Colab T4 GPU (free tier) or equivalent

Usage:
  python dsa_hidden_state_test.py --model mistral  # or gpt2, llama
"""

import torch
import numpy as np
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import gc

PHI  = (1 + 5**0.5) / 2
PHI3 = PHI**3
r_star = PHI3**0.5

MODELS = {
    'mistral': 'mistralai/Mistral-7B-v0.1',
    'llama':   'unsloth/Llama-3.2-3B',
    'gpt2':    'gpt2',
}

# 50 factual / hallucinatory sentence pairs
PAIRS = [
    ("The capital of France is Paris.",           "The capital of France is Berlin."),
    ("The capital of France is Paris.",           "The capital of France is London."),
    ("The Moon orbits the Earth.",                "The Moon orbits Jupiter."),
    ("Water boils at one hundred degrees.",       "Water boils at fifty degrees."),
    ("Gold chemical symbol is Au.",               "Gold chemical symbol is Go."),
    ("Shakespeare wrote Hamlet.",                 "Shakespeare wrote the Iliad."),
    ("Light travels faster than sound.",          "Sound travels faster than light."),
    ("Carbon dioxide is CO2.",                    "Carbon dioxide is CO3."),
    ("The Sun is center of our solar system.",    "The Moon is center of our solar system."),
    ("Oxygen has atomic number eight.",           "Oxygen has atomic number twelve."),
    ("Paris is the capital of France.",           "London is the capital of France."),
    ("DNA is deoxyribonucleic acid.",             "DNA is digital nucleic algorithm."),
    ("The Pacific is the largest ocean.",         "The Atlantic is the largest ocean."),
    ("Humans have twenty three chromosomes.",     "Humans have forty eight chromosomes."),
    ("Einstein was born in eighteen seventy nine.", "Einstein was born in nineteen fifty five."),
    ("The Earth orbits the Sun.",                 "The Sun orbits the Earth."),
    ("Penguins live in Antarctica.",              "Penguins live in the Arctic."),
    ("The heart pumps blood.",                    "The liver pumps blood."),
    ("Gravity pulls objects downward.",           "Gravity pushes objects downward."),
    ("The Amazon is in South America.",           "The Amazon is in North America."),
    ("Rome is the capital of Italy.",             "Rome is the capital of Spain."),
    ("The human body has two kidneys.",           "The human body has four kidneys."),
    ("Water is H2O.",                             "Water is H3O."),
    ("The Nile flows through Egypt.",             "The Nile flows through Brazil."),
    ("Dogs are mammals.",                         "Dogs are reptiles."),
    ("The piano has eighty eight keys.",          "The piano has sixty four keys."),
    ("Whales are mammals.",                       "Whales are fish."),
    ("The brain controls the nervous system.",    "The heart controls the nervous system."),
    ("Copper conducts electricity.",              "Wood conducts electricity."),
    ("The Sahara is a desert.",                   "The Sahara is an ocean."),
    ("Tokyo is the capital of Japan.",            "Beijing is the capital of Japan."),
    ("The sun rises in the east.",                "The sun rises in the west."),
    ("Humans breathe oxygen.",                    "Humans breathe nitrogen."),
    ("Ice melts at zero degrees Celsius.",        "Ice melts at fifty degrees Celsius."),
    ("The speed of sound is slower than light.",  "The speed of light is slower than sound."),
    ("Photosynthesis produces oxygen.",           "Photosynthesis produces nitrogen."),
    ("The Eiffel Tower is in Paris.",             "The Eiffel Tower is in Berlin."),
    ("Bats are mammals.",                         "Bats are birds."),
    ("The Atlantic separates Europe and America.", "The Pacific separates Europe and America."),
    ("Honey bees make honey.",                    "Honey bees make milk."),
    ("The moon reflects sunlight.",               "The moon generates its own light."),
    ("Diamonds are made of carbon.",              "Diamonds are made of silicon."),
    ("Blood carries oxygen to cells.",            "Blood carries nitrogen to cells."),
    ("The North Pole is in the Arctic.",          "The North Pole is in Antarctica."),
    ("Plants need sunlight to grow.",             "Plants need darkness to grow."),
    ("The human heart has four chambers.",        "The human heart has two chambers."),
    ("Sound travels through air.",                "Light travels through air only."),
    ("The Amazon river is in Brazil.",            "The Amazon river is in Australia."),
    ("Iron is magnetic.",                         "Wood is magnetic."),
    ("The Earth has one moon.",                   "The Earth has three moons."),
]


def load_model(model_key):
    model_name = MODELS[model_key]
    print(f"Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True
    )
    mdl.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  dim={mdl.config.hidden_size}, "
          f"layers={getattr(mdl.config, 'num_hidden_layers', mdl.config.n_layer)}")
    return tok, mdl


def get_stats(text, tok, mdl):
    inp = tok(text, return_tensors="pt", truncation=True,
              max_length=128).to(mdl.device)
    with torch.no_grad():
        out = mdl(**inp, output_hidden_states=True)
    result = {}
    for i, hs in enumerate(out.hidden_states):
        v     = hs[0, :, :].float().cpu().numpy()
        norms = np.linalg.norm(v, axis=1)
        result[i] = {
            'mean':  float(np.mean(norms)),
            'std':   float(np.std(norms)),
            'cv':    float(np.std(norms) / (np.mean(norms) + 1e-9)),
            'range': float(np.max(norms) - np.min(norms)),
            'dist':  float(np.mean(np.abs(norms - r_star))),
        }
    del out, inp
    return result


def run_test(model_key='mistral'):
    tok, mdl = load_model(model_key)
    n_layers = None

    print(f"\nRunning test on {len(PAIRS)} pairs...")
    all_t, all_h = [], []
    for idx, (fact, hall) in enumerate(PAIRS):
        ts = get_stats(fact, tok, mdl)
        hs = get_stats(hall, tok, mdl)
        all_t.append(ts)
        all_h.append(hs)
        if n_layers is None:
            n_layers = len(ts)
        if (idx+1) % 10 == 0:
            print(f"  {idx+1}/{len(PAIRS)} done...")

    del mdl, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics = ['mean', 'cv', 'range', 'dist']
    directions = {'mean': 'H>T', 'cv': 'T>H', 'range': 'H>T', 'dist': 'H<T'}
    pred_labels = {
        'mean':  'Hallucinatory norm LARGER (semantic escape)',
        'cv':    'Factual CV LARGER (free / noise-like)',
        'range': 'Hallucinatory range LARGER (wider spread)',
        'dist':  'Hallucinatory CLOSER to r*=√φ³ (Phi3-attractor)',
    }

    print(f"\n{'='*65}")
    print(f"RESULTS — {model_key.upper()}  |  r* = {r_star:.4f}  |  phi3 = {PHI3:.4f}")
    print(f"{'='*65}")

    for metric in metrics:
        wins = 0
        direction = directions[metric]
        print(f"\n--- {metric.upper()} | {pred_labels[metric]} ---")
        print(f"{'Layer':>6} {'T_mean':>9} {'H_mean':>9} {'Diff':>9} "
              f"{'Correct':>8} {'p':>8} {'sig':>4}")
        print("-"*60)

        for i in range(1, n_layers - 1):
            t_vals = np.array([d[i][metric] for d in all_t])
            h_vals = np.array([d[i][metric] for d in all_h])
            diff   = float(np.mean(t_vals) - np.mean(h_vals))
            correct = diff < 0 if direction in ('H>T',) else diff > 0
            if direction == 'H<T':
                correct = diff > 0  # T_dist > H_dist means H closer
            if correct:
                wins += 1
            n_c  = int(np.sum(h_vals > t_vals) if 'H>' in direction
                       else np.sum(t_vals > h_vals))
            _, p = stats.ttest_rel(t_vals, h_vals)
            sig  = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
            if sig and correct:
                print(f"{i:>6} {np.mean(t_vals):>9.4f} {np.mean(h_vals):>9.4f} "
                      f"{diff:>+9.4f} {str(correct):>8} {p:>8.5f} {sig:>4} <-- DSA")
            elif i % 5 == 0 or sig:
                print(f"{i:>6} {np.mean(t_vals):>9.4f} {np.mean(h_vals):>9.4f} "
                      f"{diff:>+9.4f} {str(correct):>8} {p:>8.5f} {sig:>4}")

        bp = stats.binomtest(wins, n_layers - 2, 0.5).pvalue
        print(f"\nCorrect direction: {wins}/{n_layers-2}  binomial p={bp:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mistral',
                        choices=['mistral', 'llama', 'gpt2'])
    args = parser.parse_args()
    run_test(args.model)

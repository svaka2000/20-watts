#!/usr/bin/env python3
"""
20 Watts — Episode 1 hardening: DOWNSTREAM TASK ACCURACY vs sparsity.

Perplexity is a proxy. A reviewer will ask: does the model still *answer
questions* when you switch off its neurons? Here we measure multiple-choice
accuracy (length-normalized continuation log-likelihood) on ARC-Easy, at each
sparsity level. Falls back to a built-in question set if ARC can't be fetched.

Run: python src/downstream_eval.py            (uses ARC-Easy, ~150 q)
     python src/downstream_eval.py --quick    (built-in set)
"""
import os, sys, json, time, argparse
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(__file__))
from sparse_patch import SparseModel

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "downstream_results.json")

# Built-in fallback set (authored; unambiguous). Used with --quick or if offline.
BUILTIN = [
    ("What gas do plants primarily absorb from the atmosphere for photosynthesis?",
     ["Oxygen", "Carbon dioxide", "Nitrogen", "Hydrogen"], 1),
    ("Which planet is closest to the Sun?", ["Venus", "Earth", "Mercury", "Mars"], 2),
    ("What is the powerhouse of the cell?",
     ["Nucleus", "Ribosome", "Mitochondria", "Golgi apparatus"], 2),
    ("Water freezes at what temperature in Celsius?", ["0", "32", "100", "-10"], 0),
    ("Which force pulls objects toward the Earth?",
     ["Magnetism", "Friction", "Gravity", "Tension"], 2),
    ("What is the chemical symbol for gold?", ["Gd", "Au", "Ag", "Go"], 1),
    ("How many legs does an insect have?", ["Six", "Eight", "Four", "Ten"], 0),
    ("Which organ pumps blood through the body?", ["Liver", "Lungs", "Heart", "Kidney"], 2),
    ("What do bees collect from flowers to make honey?",
     ["Water", "Nectar", "Pollen only", "Sap"], 1),
    ("Which is a renewable source of energy?", ["Coal", "Oil", "Solar", "Natural gas"], 2),
    ("The process by which liquid turns into gas is called?",
     ["Condensation", "Evaporation", "Freezing", "Melting"], 1),
    ("Which part of a plant conducts photosynthesis?",
     ["Roots", "Stem", "Leaves", "Flowers"], 2),
    ("What is the largest mammal on Earth?",
     ["Elephant", "Blue whale", "Giraffe", "Hippo"], 1),
    ("Sound travels fastest through which medium?",
     ["Vacuum", "Air", "Water", "Steel"], 3),
    ("Which vitamin does sunlight help the human body produce?",
     ["Vitamin A", "Vitamin C", "Vitamin D", "Vitamin K"], 2),
    ("What is the center of an atom called?",
     ["Electron", "Nucleus", "Proton cloud", "Shell"], 1),
    ("Which animal is known to change color to match its surroundings?",
     ["Frog", "Chameleon", "Owl", "Shark"], 1),
    ("What do we call animals that eat only plants?",
     ["Carnivores", "Omnivores", "Herbivores", "Decomposers"], 2),
    ("The Earth rotates on its axis once every approximately how many hours?",
     ["12", "24", "48", "365"], 1),
    ("Which gas makes up most of Earth's atmosphere?",
     ["Oxygen", "Carbon dioxide", "Nitrogen", "Argon"], 2),
    ("What simple machine is a ramp an example of?",
     ["Lever", "Pulley", "Inclined plane", "Wheel and axle"], 2),
    ("Which blood cells help fight infection?",
     ["Red blood cells", "White blood cells", "Platelets", "Plasma"], 1),
    ("What is frozen water called?", ["Steam", "Ice", "Vapor", "Dew"], 1),
    ("Which is the closest star to Earth?",
     ["Polaris", "Sirius", "The Sun", "Alpha Centauri"], 2),
]


def load_arc(n):
    from datasets import load_dataset
    ds = load_dataset("ai2_arc", "ARC-Easy", split=f"test[:{n}]")
    out = []
    for r in ds:
        labels = r["choices"]["label"]; texts = r["choices"]["text"]
        ak = r["answerKey"]
        if ak not in labels:
            continue
        out.append((r["question"], texts, labels.index(ak)))
    return out


def score_option(sm, prompt, option):
    """Length-normalized log-likelihood of `option` continuing `prompt`."""
    ctx = sm.encode(prompt)
    cont = sm.encode(" " + option.strip())
    if len(cont) == 0:
        return -1e9
    ids = mx.array([ctx + cont])
    lg = sm.logits(ids)
    logp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    seq = ctx + cont
    tgt = mx.array([seq[1:]])
    lp = mx.take_along_axis(logp[:, :-1, :], tgt[..., None], axis=-1)[0, :, 0]
    mx.eval(lp)
    lp = np.asarray(lp).astype(np.float64)
    start = len(ctx) - 1
    cont_lp = lp[start:start + len(cont)].sum()
    return float(cont_lp) / len(cont)


def accuracy(sm, questions):
    correct = 0
    for q, choices, ans in questions:
        prompt = f"Question: {q}\nAnswer:"
        scores = [score_option(sm, prompt, c) for c in choices]
        if int(np.argmax(scores)) == ans:
            correct += 1
    return correct / len(questions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()

    print("[load] model ...", flush=True)
    sm = SparseModel(verify=True)
    print(f"[load] integrity diff = {sm.max_diff:.1e}; MLP share = {sm.mlp_share:.3f}", flush=True)

    if args.quick:
        questions = BUILTIN; source = "builtin"
    else:
        try:
            questions = load_arc(args.n); source = f"ARC-Easy[:{args.n}]"
        except Exception as e:
            print("[warn] ARC unavailable, using builtin:", e, flush=True)
            questions = BUILTIN; source = "builtin"
    print(f"[data] {source}: {len(questions)} questions", flush=True)

    keeps = [1.0, 0.6, 0.5, 0.4, 0.3, 0.2]
    rows = []
    base = None
    for keep in keeps:
        sm.set_keep(keep)
        t0 = time.time()
        acc = accuracy(sm, questions)
        if base is None:
            base = acc
        rows.append({"keep": keep, "skip": round(1 - keep, 3), "accuracy": acc,
                     "delta_vs_dense": round(acc - base, 4)})
        print(f"   keep={keep:.2f} skip={1-keep:4.0%}  acc={acc:.3f}  "
              f"(Δ {acc-base:+.3f})  [{time.time()-t0:.1f}s]", flush=True)

    result = {"model": sm.model_id, "source": source, "n_questions": len(questions),
              "integrity_diff": sm.max_diff, "mlp_share": sm.mlp_share,
              "metric": "length-normalized MCQ accuracy", "sweep": rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[save] -> {os.path.abspath(OUT)}", flush=True)


if __name__ == "__main__":
    main()

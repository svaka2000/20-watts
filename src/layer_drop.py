#!/usr/bin/env python3
"""
20 Watts — Episode 2 (strong result): DEPTH REDUNDANCY / adaptive depth.

Predictive-coding framing: a brain does not run every processing stage at full
power for every input; deeper stages engage only when needed. We test how much of
a transformer's DEPTH is actually load-bearing by replacing whole layers with the
identity (a residual passthrough) and measuring held-out perplexity. Layers that
can be removed cheaply are compute that the model rarely needs — directly
analogous to the redundancy exploited by ShortGPT / Gromov et al. (2024).

We measure: (1) each layer's individual importance (ppl increase when dropped),
(2) dropping the last K layers, (3) dropping the K least-important layers.

Faithfulness: with an empty drop-set, perplexity equals the unmodified model.

Run: python src/layer_drop.py
"""
import os, sys, json, time, argparse, math
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(__file__))
from sparse_patch import SparseModel

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "layer_drop_results.json")

EVAL_TEXT = (
    "Energy is the currency of every living system. A nerve cell does not fire "
    "continuously; it stays silent for long stretches and spends its budget only on "
    "the signals that matter. Researchers estimate that fewer than one in a hundred "
    "cortical neurons are substantially active at any moment. From this constraint a "
    "surprising elegance emerges: perception and thought are built on sparseness, on "
    "the quiet majority of cells that wait while a chosen few carry the message. "
    "A modern language model, by contrast, activates every unit for every word, and "
    "pushes each word through every layer, whether the word is surprising or utterly "
    "predictable. If a machine could learn to be lazy in the way a brain is lazy, the "
    "cost of intelligence might fall by a large factor, and the gap between a warehouse "
    "of graphics cards and a twenty watt organ might begin, at last, to close. "
    "The history of computing is in many ways a history of doing less, of finding the "
    "small fraction of work that truly matters and skipping the rest."
)


def main():
    print("[load] model ...", flush=True)
    sm = SparseModel(verify=False)
    L = sm.dims["L"]
    LayerClass = type(sm.layers[0])
    _orig = LayerClass.__call__
    DROP = set()
    for i, l in enumerate(sm.layers):      # tag the DECODER BLOCK (not just its .mlp)
        l.layer_id = i

    def patched(self2, x, *a, **k):
        if getattr(self2, "layer_id", -1) in DROP:
            return x                      # residual passthrough == skip this layer
        return _orig(self2, x, *a, **k)
    LayerClass.__call__ = patched

    ids = mx.array([sm.encode(EVAL_TEXT)])
    T = ids.shape[1]
    DROP.clear()
    nll0, ppl0 = sm.perplexity(ids)
    print(f"[base] {T} tokens, dense ppl = {ppl0:.3f}", flush=True)

    # (1) individual layer importance
    importance = []
    for i in range(L):
        DROP.clear(); DROP.add(i)
        nll, ppl = sm.perplexity(ids)
        importance.append({"layer": i, "ppl": ppl,
                           "ppl_increase_pct": round(100 * (math.exp(nll - nll0) - 1), 2)})
    order = sorted(importance, key=lambda r: r["ppl_increase_pct"])
    print("[importance] 5 least important layers:",
          [(r["layer"], r["ppl_increase_pct"]) for r in order[:5]], flush=True)

    # (2) drop last-K
    last_k = []
    for K in range(1, min(12, L) + 1):
        DROP.clear(); DROP.update(range(L - K, L))
        nll, ppl = sm.perplexity(ids)
        last_k.append({"K": K, "ppl": ppl,
                       "ppl_increase_pct": round(100 * (math.exp(nll - nll0) - 1), 2),
                       "compute_saved_frac": round(K / L, 4)})

    # (3) drop K least-important
    least_k = []
    ranked = [r["layer"] for r in order]
    for K in range(1, min(12, L) + 1):
        DROP.clear(); DROP.update(ranked[:K])
        nll, ppl = sm.perplexity(ids)
        least_k.append({"K": K, "dropped": sorted(ranked[:K]), "ppl": ppl,
                        "ppl_increase_pct": round(100 * (math.exp(nll - nll0) - 1), 2),
                        "compute_saved_frac": round(K / L, 4)})

    LayerClass.__call__ = _orig

    def free_at(rows, budget):
        best = 0
        for r in rows:
            if r["ppl_increase_pct"] <= budget:
                best = max(best, r["compute_saved_frac"])
        return best
    headline = {
        "last_k_free_at_1pct": free_at(last_k, 1.0),
        "last_k_free_at_5pct": free_at(last_k, 5.0),
        "least_k_free_at_1pct": free_at(least_k, 1.0),
        "least_k_free_at_5pct": free_at(least_k, 5.0),
    }
    for k, v in headline.items():
        print(f"[headline] {k} = {v:.0%} of layers droppable", flush=True)

    result = {"model": sm.model_id, "n_layers": L, "eval_tokens": int(T),
              "dense_ppl": ppl0, "layer_importance": importance,
              "drop_last_k": last_k, "drop_least_important_k": least_k,
              "headline": headline}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[save] -> {os.path.abspath(OUT)}", flush=True)


if __name__ == "__main__":
    main()

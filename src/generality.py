#!/usr/bin/env python3
"""
20 Watts — Episode 1 generality check: does sparse firing hold on ANOTHER model?

We repeat the core sparsity/quality sweep on a different family (Llama-3.2-3B) to
show the effect is not specific to Qwen. Same bit-exact harness.

Run: python src/generality.py mlx-community/Llama-3.2-3B-Instruct-4bit
"""
import os, sys, json, math
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(__file__))
from sparse_patch import SparseModel

EVAL_TEXT = (
    "Energy is the currency of every living system. A nerve cell does not fire "
    "continuously; it stays silent for long stretches and spends its budget only on "
    "the signals that matter, because firing is expensive and the brain cannot afford "
    "to run every neuron at once. Fewer than one in a hundred cortical neurons are "
    "substantially active at any single moment, and from this constraint a surprising "
    "elegance emerges. A modern language model, by contrast, activates every one of its "
    "internal units for every word it reads, whether the word is surprising or utterly "
    "predictable, paying the full price of computation even when almost none is required. "
    "If a machine could learn to be lazy in the way a brain is lazy, the cost of "
    "intelligence might fall not by a few percent but by a large factor."
)


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Llama-3.2-3B-Instruct-4bit"
    safe = model_id.split("/")[-1]
    out = os.path.join(os.path.dirname(__file__), "..", "results", f"generality_{safe}.json")
    print(f"[load] {model_id} ...", flush=True)
    sm = SparseModel(model_id, verify=True)
    print(f"[load] integrity diff={sm.max_diff:.1e}  MLP share={sm.mlp_share:.3f}  "
          f"dims={sm.dims}", flush=True)
    ids = mx.array([sm.encode(EVAL_TEXT)])
    keeps = [1.0, 0.6, 0.5, 0.4, 0.3, 0.2]
    rows = []; nll0 = None
    for k in keeps:
        sm.set_keep(k)
        nll, ppl = sm.perplexity(ids)
        if nll0 is None:
            nll0 = nll
        rows.append({"keep": k, "skip": round(1 - k, 3), "ppl": ppl,
                     "ppl_increase_pct": round(100 * (math.exp(nll - nll0) - 1), 2)})
        print(f"   keep={k:.2f} skip={1-k:4.0%}  ppl={ppl:7.3f} "
              f"(+{rows[-1]['ppl_increase_pct']:.2f}%)", flush=True)
    free = max([r["skip"] for r in rows if r["ppl_increase_pct"] <= 1.0] + [0])
    result = {"model": model_id, "integrity_diff": sm.max_diff, "mlp_share": sm.mlp_share,
              "dims": sm.dims, "sweep": rows,
              "free_skip_at_1pct": free,
              "compute_reduction_predictor": round(free * sm.mlp_share, 4)}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[result] free skip @<1% = {free:.0%} → ~{free*sm.mlp_share:.0%} compute cut "
          f"(MLP share {sm.mlp_share:.0%})", flush=True)
    print(f"[save] -> {os.path.abspath(out)}", flush=True)


if __name__ == "__main__":
    main()

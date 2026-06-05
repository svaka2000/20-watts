#!/usr/bin/env python3
"""
20 Watts — Episode 4: SYNAPTIC PRUNING (static) vs SPARSE FIRING (dynamic).

The brain does TWO different kinds of sparsity. In development it permanently *prunes*
about half of its synapses (structural, once). And moment to moment it keeps almost all
neurons *silent* (dynamic, per input). Episode 1 measured the dynamic kind. Here we
measure the static kind on the same model — permanently removing the globally least-used
MLP neurons — and we put the two head to head.

Why the comparison matters:
  - STATIC pruning is *free to realize*: a permanently smaller weight matrix, no predictor,
    no kernel. You just ship a smaller model.
  - DYNAMIC sparsity needs a predictor (Episode 1), but should give better quality at the
    same neuron count, because the neurons a model needs CHANGE from token to token.

If dynamic clearly beats static at equal sparsity, that quantifies the *value of
adaptivity* — and explains why the brain bothers to keep pruned-down circuits dynamic.

Run: python src/synaptic_prune.py
"""
import os, sys, json, math
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(__file__))
from sparse_patch import SparseModel
from predictor import build_corpus

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "synaptic_prune_results.json")

# held-out eval passage (authored; not in the wikitext calibration set)
EVAL_TEXT = (
    "A coral reef looks like a single organism but is really a crowded city of specialists, "
    "each built for one narrow task and quiet the rest of the time. The same economy governs "
    "the body that reads this sentence. Most of its cells are idle at any instant, holding "
    "their charge in reserve, because the price of activity is paid in energy the organism "
    "cannot waste. Evolution did not arrive at this arrangement by accident; it was forced "
    "there by the relentless arithmetic of survival, in which every unnecessary expenditure "
    "is a disadvantage. A machine that hopes to match the efficiency of living things will "
    "have to learn the same lesson twice over, first by carving away the parts it never needs, "
    "and then by keeping silent the parts it needs only sometimes. The first is a permanent "
    "decision and the second is a constant one, and a truly efficient system makes both. When "
    "engineers finally take this seriously, the enormous cost of running artificial minds may "
    "shrink toward the modest budget that a biological brain has always managed to live within."
)


def main():
    print("[load] model ...", flush=True)
    sm = SparseModel(verify=True)
    L, I = sm.dims["L"], sm.dims["I"]
    print(f"[load] integrity diff={sm.max_diff:.0e}  L={L} I={I}", flush=True)

    # ---- calibration: global per-neuron importance = mean |h| over a corpus ----
    text, src = build_corpus(target_tokens=4000)
    ids = sm.encode(text)
    sums = {l: np.zeros(I, dtype=np.float64) for l in range(L)}
    counts = {"n": 0}

    def hook(lid, x, h):
        absh = mx.abs(h).reshape(-1, h.shape[-1]); mx.eval(absh)
        a = np.asarray(absh).astype(np.float64)
        sums[lid] += a.sum(0)
        if lid == L - 1:
            counts["n"] += a.shape[0]

    sm.hook = hook; sm.keep = 1.0
    pos = 0
    while pos < len(ids) and counts["n"] < 4000:
        chunk = ids[pos:pos + 512]
        if len(chunk) < 8:
            break
        out = sm.model(mx.array([chunk])); mx.eval(out)
        pos += 512
    sm.hook = None
    importance = {l: sums[l] / max(1, counts["n"]) for l in range(L)}
    print(f"[calib] {src}: {counts['n']} tokens; built global importance", flush=True)

    def static_masks(keep):
        masks = {}
        for l in range(L):
            imp = importance[l]; k = max(1, int(round(keep * I)))
            thr = np.partition(imp, I - k)[I - k]
            masks[l] = mx.array(imp >= thr)
        return masks

    eval_ids = mx.array([sm.encode(EVAL_TEXT)])
    keeps = [1.0, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    nll0, ppl0 = (None, None)
    rows = []
    for keep in keeps:
        # dynamic (per-token oracle top-k) — Episode 1's mechanism
        sm.set_keep(keep)
        nd, pd = sm.perplexity(eval_ids)
        # static (one global mask for all tokens) — Episode 4's mechanism
        masks = static_masks(keep)
        sm.mask_fn = lambda lid, h: h * masks[lid]
        ns, ps = sm.perplexity(eval_ids)
        sm.reset()
        if nll0 is None:
            nll0 = nd
        rows.append({
            "keep": keep, "skip": round(1 - keep, 3),
            "ppl_dynamic": pd, "ppl_static": ps,
            "dyn_increase_pct": round(100 * (math.exp(nd - nll0) - 1), 2),
            "stat_increase_pct": round(100 * (math.exp(ns - nll0) - 1), 2),
        })
        print(f"   skip={1-keep:4.0%}  dynamic ppl={pd:7.3f} (+{rows[-1]['dyn_increase_pct']:5.2f}%)"
              f"   static ppl={ps:7.3f} (+{rows[-1]['stat_increase_pct']:5.2f}%)", flush=True)

    # how much sparsity each method affords at <=5% quality cost
    def free(metric):
        return max([r["skip"] for r in rows if r[metric] <= 5.0] + [0])
    dyn_free, stat_free = free("dyn_increase_pct"), free("stat_increase_pct")
    result = {
        "model": sm.model_id, "calibration": src, "calib_tokens": counts["n"],
        "eval_tokens": int(eval_ids.shape[1]), "integrity_diff": sm.max_diff,
        "comparison": rows,
        "dynamic_free_skip_at_5pct": dyn_free,
        "static_free_skip_at_5pct": stat_free,
        "note": ("Static pruning is free to realize (smaller matrix, no predictor); dynamic "
                 "needs a predictor but tolerates more sparsity because the needed neurons "
                 "change per token. The gap is the measured value of adaptivity."),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[result] free skip @<=5%: dynamic {dyn_free:.0%} vs static {stat_free:.0%} "
          f"→ adaptivity buys +{(dyn_free-stat_free)*100:.0f} pts of sparsity", flush=True)
    print(f"[save] -> {os.path.abspath(OUT)}", flush=True)


if __name__ == "__main__":
    main()

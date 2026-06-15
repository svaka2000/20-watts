#!/usr/bin/env python3
"""
20 Watts — award-grade version of the conditional-vs-structural comparison.

Upgrades over src/synaptic_prune.py (which used a weak static baseline + a one-off eval):
  - STRONGER static baseline: Wanda-style neuron importance = mean|activation| × ||down-proj
    column||  (weight × activation, the SOTA-pruning principle), in addition to the
    activation-frequency baseline. Answers "your static baseline is a strawman."
  - UNIFIED held-out eval (WikiText-2 test) so dynamic and static are directly comparable
    and the baseline perplexity is the model's true ppl.
  - --model arg → run the same comparison on multiple families (generality).

Question: at equal sparsity, does dynamic per-token selection beat the BEST static pruning?

Run: python src/conditional_vs_structural.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
"""
import os, sys, json, math, argparse
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(__file__))
from sparse_patch import SparseModel
from predictor import build_corpus
from kv_eviction import build_long_text

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "results")


def downproj_colnorm(mlp, I):
    """L2 norm of each down_proj input column (the neuron's output influence)."""
    dp = mlp.down_proj
    try:
        if hasattr(dp, "scales"):
            W = mx.dequantize(dp.weight, dp.scales, dp.biases,
                              group_size=getattr(dp, "group_size", 64),
                              bits=getattr(dp, "bits", 4))
        else:
            W = dp.weight
        W = W.astype(mx.float32)                 # [out=H, in=I]
        cn = mx.sqrt((W * W).sum(axis=0))        # [I]
        mx.eval(cn)
        cn = np.asarray(cn).astype(np.float64)
        if cn.shape[0] != I:                     # safety if layout differs
            return np.ones(I)
        return cn
    except Exception as e:
        print(f"   [warn] colnorm dequant failed ({e}); using ones", flush=True)
        return np.ones(I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    args = ap.parse_args()
    keep_grid = [1.0, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

    print(f"[load] {args.model} ...", flush=True)
    sm = SparseModel(args.model, verify=True)
    L, I = sm.dims["L"], sm.dims["I"]
    print(f"[load] diff={sm.max_diff:.0e}  L={L} I={I}", flush=True)

    # ---- calibration: mean|h| per neuron + down-proj column norms ----
    calib, src = build_corpus(target_tokens=4000)
    ids = sm.encode(calib)
    sums = {l: np.zeros(I, np.float64) for l in range(L)}
    cnt = {"n": 0}

    def hook(lid, x, h):
        a = np.asarray(mx.abs(h).reshape(-1, I)).astype(np.float64)
        sums[lid] += a.sum(0)
        if lid == L - 1:
            cnt["n"] += a.shape[0]
    sm.hook = hook; sm.keep = 1.0
    pos = 0
    while pos < len(ids) and cnt["n"] < 4000:
        ch = ids[pos:pos + 512]
        if len(ch) < 8:
            break
        mx.eval(sm.model(mx.array([ch]))); pos += 512
    sm.hook = None
    mean_absh = {l: sums[l] / max(1, cnt["n"]) for l in range(L)}
    colnorm = {l: downproj_colnorm(sm.layers[l].mlp, I) for l in range(L)}
    freq_imp = mean_absh                                  # static baseline 1
    wanda_imp = {l: mean_absh[l] * colnorm[l] for l in range(L)}   # static baseline 2 (stronger)
    print(f"[calib] {src}: {cnt['n']} tokens; built freq + wanda importances", flush=True)

    def static_masks(imp, keep):
        k = max(1, int(round(keep * I)))
        out = {}
        for l in range(L):
            m = np.zeros(I, bool); m[np.argsort(-imp[l])[:k]] = True
            out[l] = mx.array(m)
        return out

    eval_text, _ = build_long_text(target_words=700)     # WikiText-2 TEST (held out)
    eval_ids = mx.array([sm.encode(eval_text)])
    T = eval_ids.shape[1]
    print(f"[eval] held-out tokens = {T}", flush=True)

    nll0 = None
    rows = []
    for keep in keep_grid:
        sm.set_keep(keep)
        nd, pd = sm.perplexity(eval_ids)
        if nll0 is None:
            nll0 = nd
        masks_f = static_masks(freq_imp, keep)
        sm.mask_fn = lambda lid, h: h * masks_f[lid]
        nf, pf = sm.perplexity(eval_ids)
        masks_w = static_masks(wanda_imp, keep)
        sm.mask_fn = lambda lid, h: h * masks_w[lid]
        nw, pw = sm.perplexity(eval_ids)
        sm.reset()
        inc = lambda nll: round(100 * (math.exp(nll - nll0) - 1), 2)
        rows.append({"keep": keep, "skip": round(1 - keep, 3),
                     "dynamic_pct": inc(nd), "static_freq_pct": inc(nf), "static_wanda_pct": inc(nw),
                     "ppl_dynamic": round(pd, 3), "ppl_static_freq": round(pf, 3),
                     "ppl_static_wanda": round(pw, 3)})
        print(f"   skip={1-keep:4.0%}  dyn {inc(nd):+6.1f}%  | static(freq) {inc(nf):+7.1f}%  "
              f"| static(Wanda) {inc(nw):+7.1f}%", flush=True)

    def free(key):
        return max([r["skip"] for r in rows if r[key] <= 5.0] + [0])
    result = {"model": args.model, "L": L, "I": I, "calib": src, "eval_tokens": int(T),
              "dense_ppl": round(math.exp(nll0), 3), "integrity_diff": sm.max_diff,
              "free_skip_at_5pct": {"dynamic": free("dynamic_pct"),
                                    "static_freq": free("static_freq_pct"),
                                    "static_wanda": free("static_wanda_pct")},
              "sweep": rows}
    safe = args.model.split("/")[-1]
    out = os.path.join(OUTDIR, f"cond_vs_struct_{safe}.json")
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(result, open(out, "w"), indent=2)
    f = result["free_skip_at_5pct"]
    print(f"\n[result] free@5%: dynamic {f['dynamic']:.0%} vs static-Wanda {f['static_wanda']:.0%} "
          f"vs static-freq {f['static_freq']:.0%}", flush=True)
    print(f"[save] -> {os.path.abspath(out)}", flush=True)


if __name__ == "__main__":
    main()

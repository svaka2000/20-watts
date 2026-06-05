#!/usr/bin/env python3
"""
20 Watts — Phase 4: THE GRAND SYNTHESIS.

Each episode targets a different brain principle and a different efficiency lever.
The whole thesis of the series is that they are INDEPENDENT and therefore STACK:

   storage      4-bit quantization        (already in the model; the 'other guy')
   computation  sparse firing  (Ep.1)     skip idle MLP neurons
   depth        adaptive depth (Ep.2)     drop redundant layers
   memory       foveated KV    (Ep.3)     attention sinks + sliding window

Here we apply them together on ONE model and measure the cumulative perplexity and
the combined compute/memory reduction. Faithfulness: the 'dense' config reproduces
the unmodified model.

Run: python src/stack_all.py
"""
import os, sys, json, math, argparse
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(__file__))
from sparse_patch import SparseModel
import mlx_lm.models.qwen2 as qwen2mod

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "stack_results.json")
LD = os.path.join(os.path.dirname(__file__), "..", "results", "layer_drop_results.json")

KV = {"on": False, "S": 4, "W": 10 ** 7}
_ORIG = qwen2mod.create_attention_mask


def kvmask(h, cache=None, *a, **k):
    if not KV["on"]:
        return _ORIG(h, cache, *a, **k)
    T = h.shape[1]
    if T <= 1:
        return None
    i = mx.arange(T)[:, None]; j = mx.arange(T)[None, :]
    allowed = (j <= i) & ((j < KV["S"]) | (j > i - KV["W"]))
    return mx.where(allowed, mx.array(0.0), mx.array(-5e4)).astype(h.dtype)


def long_text(words=900):
    try:
        from datasets import load_dataset
        for repo in ["Salesforce/wikitext", "wikitext"]:
            try:
                ds = load_dataset(repo, "wikitext-2-raw-v1", split="test")
            except Exception:
                continue
            buf, n = [], 0
            for r in ds:
                t = r["text"].strip()
                if len(t) < 40:
                    continue
                buf.append(t); n += len(t.split())
                if n > words:
                    break
            return "\n".join(buf)
    except Exception:
        pass
    return ("The brain spends energy only where it must, and so a faithful imitation "
            "must learn the same restraint across storage, computation, depth, and memory. ") * 80


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=float, default=0.6)   # sparse firing operating point
    ap.add_argument("--drop", type=int, default=4)       # layers to drop
    ap.add_argument("--window", type=int, default=512)   # KV window
    args = ap.parse_args()

    print("[load] model ...", flush=True)
    sm = SparseModel(verify=False)
    L = sm.dims["L"]
    qwen2mod.create_attention_mask = kvmask
    LayerClass = type(sm.layers[0]); _origL = LayerClass.__call__
    DROP = set()
    for i, l in enumerate(sm.layers):
        l.layer_id = i

    def patchedL(self2, x, *a, **k):
        if getattr(self2, "layer_id", -1) in DROP:
            return x
        return _origL(self2, x, *a, **k)
    LayerClass.__call__ = patchedL

    # least-important layers (from Ep.2) else fall back to last-K
    if os.path.exists(LD):
        ld = json.load(open(LD))
        ranked = [r["layer"] for r in sorted(ld["layer_importance"],
                                             key=lambda r: r["ppl_increase_pct"])]
    else:
        ranked = list(range(L - 1, -1, -1))
    drop_set = set(ranked[:args.drop])

    ids = mx.array([sm.encode(long_text())])
    T = ids.shape[1]

    def cfg(keep, drop_on, kv_on):
        sm.set_keep(keep)
        DROP.clear()
        if drop_on:
            DROP.update(drop_set)
        KV["on"] = kv_on; KV["S"] = 4; KV["W"] = args.window
        return sm.perplexity(ids)

    mlp_share = sm.mlp_share
    stages = [
        ("dense (4-bit only)", 1.0, False, False),
        (f"+ sparse firing (skip {1-args.keep:.0%})", args.keep, False, False),
        (f"+ drop {args.drop} layers", args.keep, True, False),
        (f"+ KV window {args.window}", args.keep, True, True),
    ]
    nll0 = None
    rows = []
    for name, keep, d_on, kv_on in stages:
        nll, ppl = cfg(keep, d_on, kv_on)
        if nll0 is None:
            nll0 = nll
        s = 1 - keep
        d = (args.drop / L) if d_on else 0.0
        # realizable per-token projection-compute reduction (predictor-style):
        per_layer = sm.dims["attn_flops"] + (1 - s) * sm.dims["mlp_flops"]
        dense_layer = sm.dims["attn_flops"] + sm.dims["mlp_flops"]
        comp_frac = (1 - d) * per_layer / dense_layer
        rows.append({
            "stage": name, "ppl": ppl,
            "ppl_increase_pct": round(100 * (math.exp(nll - nll0) - 1), 2),
            "compute_reduction_pct": round(100 * (1 - comp_frac), 1),
            "kv_mem_saved_pct_at_T": round(100 * max(0.0, 1 - (4 + args.window) / T), 1) if kv_on else 0.0,
        })
        print(f"   {name:34s} ppl={ppl:7.3f} (+{rows[-1]['ppl_increase_pct']:5.2f}%)  "
              f"compute -{rows[-1]['compute_reduction_pct']:.0f}%  "
              f"kvmem -{rows[-1]['kv_mem_saved_pct_at_T']:.0f}%", flush=True)

    LayerClass.__call__ = _origL
    qwen2mod.create_attention_mask = _ORIG

    final = rows[-1]
    summary = {
        "model": sm.model_id, "eval_tokens": int(T), "operating_point":
            {"sparse_keep": args.keep, "layers_dropped": args.drop, "kv_window": args.window},
        "stages": rows,
        "headline": {
            "final_ppl_increase_pct": final["ppl_increase_pct"],
            "compute_reduction_pct": final["compute_reduction_pct"],
            "kv_mem_saved_pct_at_T": final["kv_mem_saved_pct_at_T"],
            "plus_4bit_storage": "4x smaller weights (orthogonal)",
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[stack] compute -{final['compute_reduction_pct']:.0f}% + KV mem "
          f"-{final['kv_mem_saved_pct_at_T']:.0f}% at +{final['ppl_increase_pct']:.1f}% ppl, "
          f"on top of 4-bit", flush=True)
    print(f"[save] -> {os.path.abspath(OUT)}", flush=True)


if __name__ == "__main__":
    main()

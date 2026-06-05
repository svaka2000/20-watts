#!/usr/bin/env python3
"""
20 Watts — Episode 3: FOVEATED MEMORY (KV-cache eviction).

Brain principle: you do not store a verbatim transcript of every word you hear.
You keep the gist plus the most recent detail; older specifics are compressed or
dropped. A transformer, in contrast, keeps a perfect key/value cache for EVERY
past token, so memory and attention cost grow without bound.

We test a brain-like memory: keep only a few "attention sink" tokens at the start
(Xiao et al., StreamingLLM, 2023) plus a sliding window of the most recent W
tokens, and drop the rest. We measure held-out perplexity over a long passage as a
function of the kept budget (S + W), which equals the KV-cache memory retained.

Faithfulness check: with a large window the streaming mask reduces to plain causal
attention, and perplexity must match the unmodified model (our integrity test).

Run: python src/kv_eviction.py
"""
import os, sys, json, time, argparse, math
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(__file__))
from sparse_patch import SparseModel
import mlx_lm.models.qwen2 as qwen2mod

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "kv_eviction_results.json")

_STATE = {"S": 4, "W": 100000, "on": False}
_ORIG_MASK = qwen2mod.create_attention_mask


def streaming_mask(h, cache=None, *a, **k):
    if not _STATE["on"]:
        return _ORIG_MASK(h, cache, *a, **k)
    T = h.shape[1]
    if T <= 1:
        return None
    S, W = _STATE["S"], _STATE["W"]
    i = mx.arange(T)[:, None]
    j = mx.arange(T)[None, :]
    allowed = (j <= i) & ((j < S) | (j > i - W))
    return mx.where(allowed, mx.array(0.0), mx.array(-5e4)).astype(h.dtype)


def build_long_text(target_words=1600):
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
                if n > target_words:
                    break
            return "\n".join(buf), repo
    except Exception:
        pass
    base = ("Memory in living systems is selective. The hippocampus does not record "
            "every detail of an experience; it preserves the gist and a few salient "
            "anchors, allowing the rest to fade. This is why you recall the shape of a "
            "conversation but not its exact words. ")
    return base * 60, "builtin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="32,64,128,256,512")
    ap.add_argument("--sinks", type=int, default=4)
    args = ap.parse_args()

    print("[load] model ...", flush=True)
    sm = SparseModel(verify=False)
    qwen2mod.create_attention_mask = streaming_mask    # install global mask hook

    text, src = build_long_text()
    ids = mx.array([sm.encode(text)])
    T = ids.shape[1]
    print(f"[data] {src}: {T} tokens", flush=True)

    # baseline (true causal)
    _STATE["on"] = False
    nll0, ppl0 = sm.perplexity(ids)
    # faithfulness: our mask with huge window == causal
    _STATE["on"] = True; _STATE["S"] = args.sinks; _STATE["W"] = 10 ** 7
    nll_chk, ppl_chk = sm.perplexity(ids)
    faith = abs(ppl_chk - ppl0)
    print(f"[check] causal ppl={ppl0:.3f}; streaming(W=inf) ppl={ppl_chk:.3f}; "
          f"|Δ|={faith:.2e}", flush=True)

    rows = []
    for W in [int(x) for x in args.windows.split(",")]:
        _STATE["on"] = True; _STATE["W"] = W; _STATE["S"] = args.sinks
        nll, ppl = sm.perplexity(ids)
        budget = args.sinks + W
        rows.append({"sinks": args.sinks, "window": W, "kept_tokens": budget,
                     "mem_kept_frac_at_T": round(min(1.0, budget / T), 4),
                     "mem_saved_frac_at_T": round(max(0.0, 1 - budget / T), 4),
                     "ppl": ppl, "ppl_increase_pct": round(100 * (math.exp(nll - nll0) - 1), 3)})
        print(f"   sinks={args.sinks} W={W:4d} keep={budget:5d}/{T} "
              f"({max(0,1-budget/T):.0%} mem saved)  ppl={ppl:.3f} "
              f"(+{rows[-1]['ppl_increase_pct']:.2f}%)", flush=True)

    qwen2mod.create_attention_mask = _ORIG_MASK        # restore
    result = {"model": sm.model_id, "corpus": src, "eval_tokens": int(T),
              "causal_ppl": ppl0, "faithfulness_abs_ppl_diff": faith,
              "method": "attention sinks + sliding window (StreamingLLM-style)",
              "sweep": rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[save] -> {os.path.abspath(OUT)}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
20 Watts — Episode 1: honest wall-clock latency (MLX prefill), dense vs sparse.

This is the reality check on the FLOP story. We time a fixed prefill at keep=1.0
vs keep=0.5 on MLX. The expected — and honest — finding is that NAIVE oracle top-k
(a full sort over 18,944 neurons per token) is NOT faster, because the sort costs
more than the matmul it saves at this scale. That is precisely WHY realizing the
savings needs a cheap predictor or a fused top-k kernel (Deja Vu's contribution),
not a sort. We report the measured numbers, good or bad.
"""
import os, sys, time, json
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(__file__))
from sparse_patch import SparseModel

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "latency_results.json")
TEXT = ("A language model spends most of its energy multiplying numbers that turn out "
        "not to matter, and the central question of efficient inference is how to avoid "
        "that wasted work without harming the answer. ") * 30


def main():
    sm = SparseModel(verify=False)
    ids = mx.array([sm.encode(TEXT)])
    T = ids.shape[1]
    print(f"[latency] prefill of {T} tokens", flush=True)

    def bench(keep, reps=6):
        sm.set_keep(keep)
        for _ in range(2):
            mx.eval(sm.model(ids))                 # warmup / compile
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            o = sm.model(ids); mx.eval(o)
            ts.append(time.perf_counter() - t0)
        return float(np.min(ts)), float(np.median(ts))

    d_min, d_med = bench(1.0)
    s_min, s_med = bench(0.5)
    result = {
        "model": sm.model_id, "prefill_tokens": int(T),
        "dense_min_s": round(d_min, 4), "dense_median_s": round(d_med, 4),
        "sparse_keep0.5_min_s": round(s_min, 4), "sparse_keep0.5_median_s": round(s_med, 4),
        "dense_tok_per_s": round(T / d_min, 1), "sparse_tok_per_s": round(T / s_min, 1),
        "naive_topk_speedup": round(d_min / s_min, 3),
        "interpretation": ("Oracle top-k uses a full per-token sort; if speedup <= 1 the "
                           "sort dominates the saved matmul, which is exactly why a cheap "
                           "predictor or fused top-k kernel (Deja Vu) is required to turn "
                           "the ~52% FLOP headroom into wall-clock speedup."),
    }
    print(f"   dense  : {result['dense_tok_per_s']} tok/s", flush=True)
    print(f"   sparse : {result['sparse_tok_per_s']} tok/s  (naive top-k speedup "
          f"{result['naive_topk_speedup']}×)", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[save] -> {os.path.abspath(OUT)}", flush=True)


if __name__ == "__main__":
    main()

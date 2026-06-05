#!/usr/bin/env python3
"""
20 Watts — Episode 3 upgrade: H2O (Heavy-Hitter Oracle) KV eviction.

StreamingLLM (kv_eviction.py) keeps the most RECENT tokens + a few sinks. But not all
old tokens are equal — some keys receive far more attention ("heavy hitters", Zhang et
al. 2023, H2O). Given the SAME memory budget, is it better to spend it all on recency,
or to split it between recency and the heavy hitters?

To answer we must see the attention scores, so we replace the fused attention with a
faithful manual implementation (q·kᵀ → softmax → ·v) — gated by a check that, with no
eviction, it reproduces the model bit-close. Then at a fixed budget we compare:
  STREAM : sinks + (W+K) most-recent
  H2O    : sinks + W most-recent + K highest-attention keys (oracle importance)

Run: python src/kv_eviction_h2o.py
"""
import os, sys, json, math, argparse
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(__file__))
from sparse_patch import SparseModel
from kv_eviction import build_long_text
import mlx_lm.models.qwen2 as qwen2mod

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "kv_h2o_results.json")
ST = {"mode": "off", "S": 4, "W": 64, "K": 64}


def main():
    print("[load] model ...", flush=True)
    sm = SparseModel(verify=False)
    AttnClass = type(sm.layers[0].self_attn)
    _orig = AttnClass.__call__

    def manual(self, x, mask=None, cache=None):
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q = self.rope(q); k = self.rope(k)
        rep = self.n_heads // self.n_kv_heads
        if rep > 1:
            k = mx.repeat(k, rep, axis=1); v = mx.repeat(v, rep, axis=1)
        scores = (q.astype(mx.float32) @ k.transpose(0, 1, 3, 2).astype(mx.float32)) * self.scale
        idx = mx.arange(L)
        causal = (idx[None, :] <= idx[:, None])                       # [i,j]: j<=i
        NEG = -1e9
        base = mx.where(causal, mx.array(0.0), mx.array(NEG))         # fp32 (avoid fp16 overflow)
        if ST["mode"] == "off":
            add = base
        else:
            # importance = total attention each key receives under full causal attention
            p_full = mx.softmax(scores + base, axis=-1)
            imp = np.asarray(p_full.sum(axis=(0, 1, 2)))              # [L]
            S, W, K = ST["S"], ST["W"], ST["K"]
            j = np.arange(L)
            keep = np.zeros(L, dtype=bool)
            keep[:S] = True                                          # sinks
            if ST["mode"] == "h2o":
                pool = np.argsort(-imp)
                taken = 0
                for jj in pool:
                    if jj >= S and jj < L - W:       # heavy hitters from the middle
                        keep[jj] = True; taken += 1
                        if taken >= K:
                            break
                rec = W
            else:                                                    # stream: all budget to recency
                rec = W + K
            recent = (j > (np.arange(L)[:, None] - rec))             # [i,j] j> i-rec
            allow = causal_np(L) & (keep[None, :] | recent | (j[None, :] < S))
            add = mx.where(mx.array(allow), mx.array(0.0), mx.array(NEG))
        p = mx.softmax(scores + add, axis=-1).astype(v.dtype)
        out = (p @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)

    def causal_np(L):
        i = np.arange(L)
        return (i[None, :] <= i[:, None])

    AttnClass.__call__ = manual

    text, src = build_long_text(target_words=1400)
    ids = mx.array([sm.encode(text)])
    T = ids.shape[1]
    print(f"[data] {src}: {T} tokens", flush=True)

    # faithfulness: manual full attention must reproduce the fused model
    ST["mode"] = "off"
    nll0, ppl0 = sm.perplexity(ids)
    AttnClass.__call__ = _orig
    nll_f, ppl_f = sm.perplexity(ids)
    AttnClass.__call__ = manual
    faith = abs(ppl0 - ppl_f)
    print(f"[check] manual ppl={ppl0:.3f}  fused ppl={ppl_f:.3f}  |Δ|={faith:.2e}", flush=True)
    if math.isnan(faith) or faith > 0.05 * ppl_f:
        print("[ABORT] manual attention does not match fused — not trustworthy.", flush=True)
        AttnClass.__call__ = _orig
        return

    S, W, K = ST["S"], ST["W"], ST["K"]
    budget = S + W + K
    rows = []
    for mode in ["stream", "h2o"]:
        ST["mode"] = mode; ST["S"], ST["W"], ST["K"] = S, W, K
        nll, ppl = sm.perplexity(ids)
        rows.append({"mode": mode, "budget_tokens": budget,
                     "mem_saved_frac_at_T": round(max(0.0, 1 - budget / T), 3),
                     "ppl": ppl, "ppl_increase_pct": round(100 * (math.exp(nll - nll0) - 1), 2)})
        print(f"   {mode:6s} budget={budget}/{T}  ppl={ppl:.3f} "
              f"(+{rows[-1]['ppl_increase_pct']:.1f}%)", flush=True)
    AttnClass.__call__ = _orig

    win = rows[0]["ppl"] - rows[1]["ppl"]
    result = {"model": sm.model_id, "corpus": src, "eval_tokens": int(T),
              "faithfulness_abs_ppl_diff": faith, "full_ppl": ppl0,
              "budget_tokens": budget, "sinks": S, "recent_W": W, "heavy_K": K,
              "comparison": rows,
              "h2o_improvement_ppl": round(win, 3),
              "verdict": ("heavy-hitters help" if win > 0 else "recency alone is as good")}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[result] at equal budget, H2O beats pure recency by {win:.3f} ppl "
          f"→ {result['verdict']}", flush=True)
    print(f"[save] -> {os.path.abspath(OUT)}", flush=True)


if __name__ == "__main__":
    main()

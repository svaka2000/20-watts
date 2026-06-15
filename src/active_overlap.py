#!/usr/bin/env python3
"""
20 Watts — the MECHANISM behind conditional > structural sparsity.

Episode 4 showed dynamic per-token sparsity tolerates ~2x the neuron removal of static
pruning. This script asks WHY, with a direct measurement: how input-dependent is the set
of active MLP neurons? If a single fixed ("static") set could serve every token, static
pruning would match dynamic. We quantify the gap three ways, per layer, at keep=0.5:

  (1) mean_static_capture — the BEST fixed 50% set (top neurons by activation frequency)
      captures what fraction of the *average token's* actually-active neurons. Dynamic
      captures 100% by construction; this number is the ceiling for any static method.
  (2) mean_pairwise_jaccard — overlap of the active set between two random tokens
      (1.0 = identical needs, 0 = disjoint needs).
  (3) frac_ever_active — fraction of neurons that are in the active half for at least one
      token (how much of the network the inputs collectively demand).

Low capture / low Jaccard = the active set is highly input-dependent = static pruning
*cannot* match dynamic. This is the causal explanation for the Episode 4 result.

Run: python src/active_overlap.py
"""
import os, sys, json
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(__file__))
from sparse_patch import SparseModel
from predictor import build_corpus

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "active_overlap_results.json")


def main():
    keep = 0.5
    print("[load] model ...", flush=True)
    sm = SparseModel(verify=True)
    L, I = sm.dims["L"], sm.dims["I"]
    k = max(1, int(round(keep * I)))
    print(f"[load] diff={sm.max_diff:.0e}  L={L} I={I}  keep={keep} (k={k})", flush=True)

    text, src = build_corpus(target_tokens=4000)
    ids = sm.encode(text)
    masks = {l: [] for l in range(L)}

    def hook(lid, x, h):
        absh = mx.abs(h)
        thr = mx.sort(absh, axis=-1)[..., I - k:I - k + 1]
        m = (absh >= thr).reshape(-1, I)
        mx.eval(m)
        masks[lid].append(np.asarray(m).astype(bool))

    sm.hook = hook; sm.keep = 1.0
    n_tok = 2000
    pos = 0
    while pos < len(ids) and sum(a.shape[0] for a in masks[0]) < n_tok:
        ch = ids[pos:pos + 512]
        if len(ch) < 8:
            break
        mx.eval(sm.model(mx.array([ch]))); pos += 512
    sm.hook = None
    got = sum(a.shape[0] for a in masks[0])
    print(f"[collect] {src}: {got} tokens × {L} layers", flush=True)

    rng = np.random.default_rng(0)
    rows = []
    for l in range(L):
        M = np.concatenate(masks[l], 0)[:n_tok]            # [N, I] bool
        N = M.shape[0]
        freq = M.mean(0)                                    # activation frequency per neuron
        static_set = np.zeros(I, bool)
        static_set[np.argsort(-freq)[:k]] = True            # best fixed 50% set
        active_counts = np.maximum(1, M.sum(1))
        capture = (M & static_set[None, :]).sum(1) / active_counts
        a = rng.integers(0, N, 3000); b = rng.integers(0, N, 3000)
        inter = (M[a] & M[b]).sum(1); union = np.maximum(1, (M[a] | M[b]).sum(1))
        rows.append({
            "layer": l,
            "mean_static_capture": round(float(capture.mean()), 4),
            "mean_pairwise_jaccard": round(float((inter / union).mean()), 4),
            "frac_ever_active": round(float(M.any(0).mean()), 4),
        })
        masks[l] = None
        print(f"   L{l:2d}: static captures {rows[-1]['mean_static_capture']*100:4.1f}% of each "
              f"token's active set | jaccard {rows[-1]['mean_pairwise_jaccard']:.3f}", flush=True)

    mc = float(np.mean([r["mean_static_capture"] for r in rows]))
    jc = float(np.mean([r["mean_pairwise_jaccard"] for r in rows]))
    ea = float(np.mean([r["frac_ever_active"] for r in rows]))
    result = {
        "model": sm.model_id, "keep": keep, "n_tokens": got, "corpus": src,
        "mean_static_capture": round(mc, 4),
        "mean_pairwise_jaccard": round(jc, 4),
        "mean_frac_ever_active": round(ea, 4),
        "interpretation": (
            f"The best fixed 50% set captures only {mc*100:.0f}% of the average token's active "
            f"neurons (per-token selection captures 100%). Two random tokens share only "
            f"{jc*100:.0f}% of their active set (Jaccard). {ea*100:.0f}% of neurons are active "
            f"for at least one token. The active set is highly input-dependent, which is WHY "
            f"dynamic sparsity preserves quality and a single static pruning cannot."),
        "per_layer": rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[result] best static set captures {mc*100:.1f}% of each token's needs; "
          f"cross-token Jaccard {jc:.3f}", flush=True)
    print(f"[save] -> {os.path.abspath(OUT)}", flush=True)


if __name__ == "__main__":
    main()

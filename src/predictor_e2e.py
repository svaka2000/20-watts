#!/usr/bin/env python3
"""
20 Watts — Episode 1, the definitive realizability test: END-TO-END predicted masking.

predictor.py showed a cheap predictor recovers most of the active *mass* per layer. But
the real question is what happens when you use PREDICTED masks on ALL layers at once, and
let any errors compound through depth. Here we train a low-rank predictor for every layer,
then run the full model with predicted neuron-skipping everywhere and compare held-out
perplexity to (a) the dense model and (b) the ORACLE per-token top-k (the headroom).

If predicted ≈ oracle, the 52% is realizable end-to-end with a trivial predictor.
If predicted ≫ oracle, prediction errors compound and a stronger predictor is needed.
Either way we report it honestly.

Run: python src/predictor_e2e.py
"""
import os, sys, json, time, math
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(__file__))
from sparse_patch import SparseModel
from predictor import build_corpus, train_predictor

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "predictor_e2e_results.json")

EVAL_TEXT = (
    "Efficiency in nature is never an afterthought; it is the constraint from which form "
    "itself is derived. A river finds the path of least resistance, a vine climbs only "
    "where there is light, and a nervous system spends its energy only where a decision "
    "must be made. The machines we build to imitate thought have so far ignored this "
    "discipline, computing everything everywhere at once, and the bill for that extravagance "
    "is now coming due in megawatts. The remedy is not a single clever trick but a habit of "
    "restraint learned from biology, applied at every level at which a system can choose to "
    "do less without knowing less."
)


def main():
    keep = 0.5
    print("[load] model ...", flush=True)
    sm = SparseModel(verify=True)
    L, I, H = sm.dims["L"], sm.dims["I"], sm.dims["H"]
    k = max(1, int(round(keep * I)))
    print(f"[load] diff={sm.max_diff:.0e}  L={L} I={I}  keep={keep} (k={k})", flush=True)

    # ---- collect (x, oracle top-k mask) for EVERY layer ----
    text, src = build_corpus(target_tokens=5000)
    ids = sm.encode(text)
    store = {l: {"x": [], "y": []} for l in range(L)}

    def hook(lid, x, h):
        absh = mx.abs(h)
        thr = mx.sort(absh, axis=-1)[..., I - k:I - k + 1]
        ym = (absh >= thr)
        xf = x.reshape(-1, H); yf = ym.reshape(-1, I)
        mx.eval(xf, yf)
        store[lid]["x"].append(np.asarray(xf).astype(np.float16))
        store[lid]["y"].append(np.asarray(yf).astype(np.uint8))

    sm.hook = hook; sm.keep = 1.0
    n_tok = 4000
    pos = 0
    while pos < len(ids) and sum(a.shape[0] for a in store[0]["x"]) < n_tok:
        ch = ids[pos:pos + 512]
        if len(ch) < 8:
            break
        mx.eval(sm.model(mx.array([ch]))); pos += 512
    sm.hook = None
    got = sum(a.shape[0] for a in store[0]["x"])
    print(f"[collect] {src}: {got} tokens/layer for all {L} layers", flush=True)

    # ---- train a predictor per layer ----
    preds = {}
    recalls = []
    t0 = time.time()
    for l in range(L):
        X = np.concatenate(store[l]["x"], 0)[:n_tok]
        Y = np.concatenate(store[l]["y"], 0)[:n_tok]
        m, recall, _, _ = train_predictor(X, Y, H, I, r=512, steps=300, n_eval=500)
        preds[l] = m; recalls.append(recall)
        store[l] = None
    print(f"[train] {L} predictors in {time.time()-t0:.0f}s; mean recall {np.mean(recalls):.3f}", flush=True)

    eval_ids = mx.array([sm.encode(EVAL_TEXT)])

    # dense
    sm.reset()
    nll0, ppl0 = sm.perplexity(eval_ids)
    # oracle per-token top-k (the headroom)
    sm.set_keep(keep)
    nllo, pplo = sm.perplexity(eval_ids)
    # predicted masks on every layer
    pmask = {}

    def hook_pred(lid, x, h):
        sc = preds[lid](x.astype(mx.float32))
        thr = mx.sort(sc, axis=-1)[..., I - k:I - k + 1]
        pmask[lid] = (sc >= thr)
    sm.reset()
    sm.hook = hook_pred
    sm.mask_fn = lambda lid, h: h * pmask[lid].astype(h.dtype)
    nllp, pplp = sm.perplexity(eval_ids)
    sm.reset()

    def inc(nll):
        return round(100 * (math.exp(nll - nll0) - 1), 2)
    result = {
        "model": sm.model_id, "keep": keep, "corpus": src, "n_train_tokens": got,
        "eval_tokens": int(eval_ids.shape[1]), "mean_layer_recall": round(float(np.mean(recalls)), 4),
        "dense_ppl": ppl0,
        "oracle_topk_ppl": pplo, "oracle_increase_pct": inc(nllo),
        "predicted_ppl": pplp, "predicted_increase_pct": inc(nllp),
        "gap_pred_minus_oracle_pct": round(inc(nllp) - inc(nllo), 2),
        "verdict": ("realizable end-to-end (predicted ≈ oracle)"
                    if inc(nllp) - inc(nllo) < 5 else
                    "partly realizable — errors compound, stronger predictor needed"),
    }
    print(f"[result] dense {ppl0:.2f} | oracle50% {pplo:.2f} (+{inc(nllo)}%) | "
          f"predicted50% {pplp:.2f} (+{inc(nllp)}%)", flush=True)
    print(f"[result] {result['verdict']}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[save] -> {os.path.abspath(OUT)}", flush=True)


if __name__ == "__main__":
    main()

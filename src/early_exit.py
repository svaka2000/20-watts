#!/usr/bin/env python3
"""
20 Watts — Episode 2: PREDICTIVE CODING (early exit / adaptive depth).

Brain principle: the cortex does not re-process information it can already
predict; it spends energy chiefly on *prediction error* / surprise
(Rao & Ballard, 1999). A dense transformer, by contrast, pushes every token
through all L layers — even when the answer is obvious after a few.

We measure how early the model's next-token prediction *stabilizes*, using the
"logit lens": apply the final norm + output head to each layer's hidden state to
read out what the model would predict if it stopped there. For each token we find
the earliest layer whose top-1 matches the full-depth top-1 (and stays matched),
and we evaluate a confidence-thresholded early-exit rule (cf. CALM, Schuster 2022).

Output: average exit depth, compute saved, and top-1 agreement with full model.
This measures the HEADROOM for adaptive depth (true early exit needs per-layer
heads/calibration; logit-lens is the honest upper bound on stabilization).

Run: python src/early_exit.py
"""
import os, sys, json, time, argparse
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from sparse_patch import SparseModel

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "early_exit_results.json")

EVAL_TEXT = (
    "The human brain is often described as the most complex object in the known "
    "universe, yet it runs on roughly the power of a dim light bulb. Scientists who "
    "study energy in the nervous system have found that this efficiency comes not from "
    "raw speed but from restraint. Neurons fire sparingly, signals are predicted before "
    "they arrive, and memory is stored as compressed impressions rather than exact "
    "recordings. When a sound is expected, the brain barely reacts; when it is "
    "surprising, attention floods the relevant circuits. This is the logic of "
    "predictive coding, and it suggests that intelligence is less about computing more "
    "and more about computing only what is necessary. A machine built on the same "
    "principle would think hard about the difficult parts of a sentence and coast "
    "through the easy ones, spending its energy where the meaning actually lives."
)


def get_head_and_norm(model):
    """Return (head_fn, norm_fn) for the logit lens, handling tied/untied heads."""
    inner = model.model
    norm = inner.norm
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:
        head = lm_head
    else:
        emb = inner.embed_tokens
        head = lambda h: emb.as_linear(h)
    return head, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confidences", default="0.5,0.7,0.9,0.95")
    args = ap.parse_args()

    print("[load] model ...", flush=True)
    sm = SparseModel(verify=False)        # no MLP masking here
    L = sm.dims["L"]
    head, norm = get_head_and_norm(sm.model)

    # capture each decoder layer's output hidden state
    LayerClass = type(sm.layers[0])
    _origL = LayerClass.__call__
    captured = []

    def patchedL(self2, *a, **k):
        out = _origL(self2, *a, **k)
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h)
        return out
    LayerClass.__call__ = patchedL

    ids = mx.array([sm.encode(EVAL_TEXT)])
    T = ids.shape[1]
    print(f"[run] forward over {T} tokens, L={L} layers ...", flush=True)
    captured.clear()
    full_logits = sm.model(ids)
    mx.eval(full_logits)
    LayerClass.__call__ = _origL          # unpatch

    final_pred = np.asarray(mx.argmax(full_logits, axis=-1))[0]   # [T]

    # logit-lens prediction + confidence at each layer
    per_layer_pred = np.zeros((L, T), dtype=np.int32)
    per_layer_conf = np.zeros((L, T), dtype=np.float32)
    for l in range(L):
        h = captured[l]
        logits_l = head(norm(h))
        probs = mx.softmax(logits_l, axis=-1)
        conf = mx.max(probs, axis=-1)
        pred = mx.argmax(logits_l, axis=-1)
        mx.eval(pred, conf)
        per_layer_pred[l] = np.asarray(pred)[0]
        per_layer_conf[l] = np.asarray(conf)[0]

    # earliest layer where top-1 matches final AND stays matched to the end
    stabilize = np.full(T, L, dtype=np.int32)
    for t in range(T):
        for l in range(L):
            if per_layer_pred[l, t] == final_pred[t] and np.all(per_layer_pred[l:, t] == final_pred[t]):
                stabilize[t] = l + 1     # 1-indexed depth used
                break
    avg_stab = float(stabilize.mean())

    # confidence-thresholded early exit
    conf_rows = []
    for tau in [float(x) for x in args.confidences.split(",")]:
        exit_layer = np.full(T, L, dtype=np.int32)
        exit_pred = final_pred.copy()
        for t in range(T):
            for l in range(L):
                if per_layer_conf[l, t] >= tau:
                    exit_layer[t] = l + 1
                    exit_pred[t] = per_layer_pred[l, t]
                    break
        agree = float((exit_pred == final_pred).mean())
        avg_depth = float(exit_layer.mean())
        conf_rows.append({"tau": tau, "avg_exit_depth": round(avg_depth, 2),
                          "layers_saved_frac": round(1 - avg_depth / L, 4),
                          "top1_agreement": round(agree, 4)})
        print(f"   τ={tau:.2f}: avg depth {avg_depth:.1f}/{L} "
              f"({(1-avg_depth/L):.0%} saved), top-1 agree {agree:.3f}", flush=True)

    result = {"model": sm.model_id, "n_layers": L, "eval_tokens": int(T),
              "method": "logit-lens stabilization + confidence early-exit",
              "avg_stabilization_depth": round(avg_stab, 2),
              "stabilization_layers_saved_frac": round(1 - avg_stab / L, 4),
              "confidence_sweep": conf_rows,
              "note": "Logit-lens is an upper bound on adaptive-depth headroom; "
                      "true early exit needs per-layer heads (CALM)."}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[result] mean stabilization depth = {avg_stab:.1f}/{L} "
          f"→ {(1-avg_stab/L):.0%} of layers are 'wasted' on the average token", flush=True)
    print(f"[save] -> {os.path.abspath(OUT)}", flush=True)


if __name__ == "__main__":
    main()

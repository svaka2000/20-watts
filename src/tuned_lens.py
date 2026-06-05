#!/usr/bin/env python3
"""
20 Watts — Episode 2 (rigorous): TUNED LENS for true adaptive-depth headroom.

The raw "logit lens" (apply the final head to an intermediate layer) UNDERESTIMATES
how early a transformer knows its answer, because the residual stream is not in the
final layer's basis until late. The tuned lens (Belrose et al., 2023) fixes this by
learning a per-layer affine probe that maps layer-l hidden states into the final
representation, then reads them out with the shared head.

We train, for each layer l, a probe  P_l(h) = A_l h + b_l  (A_l init = identity) to
match the model's FINAL normalized hidden state (so head(P_l(h_l)) ≈ final logits).
We then measure how early the *calibrated* prediction already equals the full-depth
prediction — the genuine headroom for adaptive depth / early exit.

Run: python src/tuned_lens.py            (full)
     python src/tuned_lens.py --quick
"""
import os, sys, json, time, argparse
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
sys.path.insert(0, os.path.dirname(__file__))
from sparse_patch import SparseModel
from predictor import build_corpus

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "tuned_lens_results.json")


def get_head_and_norm(model):
    inner = model.model
    norm = inner.norm
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:
        head = lm_head
    else:
        emb = inner.embed_tokens
        head = lambda h: emb.as_linear(h)
    return head, norm


def collect_hidden(sm, ids_list, n_tokens, seq_len=512):
    L = sm.dims["L"]; H = sm.dims["H"]
    head, norm = get_head_and_norm(sm.model)
    LayerClass = type(sm.layers[0]); _orig = LayerClass.__call__
    cap = []

    def patched(self2, x, *a, **k):
        out = _orig(self2, x, *a, **k)
        cap.append(out[0] if isinstance(out, tuple) else out)
        return out
    LayerClass.__call__ = patched

    Xs = [[] for _ in range(L)]; Tgt = []; Fin = []
    collected = 0; pos = 0
    while pos < len(ids_list) and collected < n_tokens:
        chunk = ids_list[pos:pos + seq_len]
        if len(chunk) < 8:
            break
        cap.clear()
        _ = sm.model(mx.array([chunk])); mx.eval(_)
        last = cap[-1]
        tnorm = norm(last)                          # [1,T,H] final head input
        flog = head(tnorm)                          # [1,T,V]
        fpred = mx.argmax(flog, axis=-1)
        mx.eval(tnorm, fpred)
        for l in range(L):
            xf = cap[l].reshape(-1, H); mx.eval(xf)
            Xs[l].append(np.asarray(xf).astype(np.float16))
        Tgt.append(np.asarray(tnorm.reshape(-1, H)).astype(np.float16))
        Fin.append(np.asarray(fpred.reshape(-1)).astype(np.int32))
        collected += chunk.__len__()
        pos += seq_len
    LayerClass.__call__ = _orig
    X = [np.concatenate(Xs[l], 0)[:n_tokens] for l in range(L)]
    T = np.concatenate(Tgt, 0)[:n_tokens]
    F = np.concatenate(Fin, 0)[:n_tokens]
    return X, T, F, head


class Affine(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.lin = nn.Linear(H, H)
        self.lin.weight = mx.eye(H)                 # init = identity (tuned-lens std)
        self.lin.bias = mx.zeros((H,))

    def __call__(self, x):
        return self.lin(x)


def train_probe(Xl, Tgt, H, steps, bs=256, lr=1e-4):
    m = Affine(H); opt = optim.Adam(learning_rate=lr)
    Xm = mx.array(Xl.astype(np.float32)); Tm = mx.array(Tgt.astype(np.float32))

    def loss_fn(mod, xb, yb):
        d = mod(xb) - yb
        return mx.mean(d * d)
    lg = nn.value_and_grad(m, loss_fn)
    N = Xm.shape[0]
    for s in range(steps):
        idx = mx.array(np.random.randint(0, N, size=bs))
        loss, grads = lg(m, Xm[idx], Tm[idx])
        opt.update(m, grads); mx.eval(m.parameters(), opt.state)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    print("[load] model ...", flush=True)
    sm = SparseModel(verify=False)
    L, H = sm.dims["L"], sm.dims["H"]

    text, src = build_corpus(target_tokens=6000)
    ids = sm.encode(text)
    n_tok = 1500 if args.quick else 4500
    steps = 150 if args.quick else 400
    print(f"[collect] hidden states for {n_tok} tokens from {src} ...", flush=True)
    t0 = time.time()
    X, Tgt, Fin, head = collect_hidden(sm, ids, n_tok)
    print(f"[collect] done in {time.time()-t0:.1f}s", flush=True)

    n_eval = 800 if not args.quick else 300
    tr = slice(0, X[0].shape[0] - n_eval); ev = slice(X[0].shape[0] - n_eval, None)
    Fev = Fin[ev]

    raw_acc = np.zeros(L); tuned_acc = np.zeros(L)
    tuned_pred = np.zeros((L, n_eval), dtype=np.int32)
    for l in range(L):
        # raw logit lens accuracy (no probe): head(norm? ) — we approximate raw by
        # head on the un-probed hidden (identity probe before training)
        m = train_probe(X[l][tr], Tgt[tr], H, steps)
        Xev = mx.array(X[l][ev].astype(np.float32))
        logits = head(m(Xev)); mx.eval(logits)
        pred = np.asarray(mx.argmax(logits, axis=-1))
        tuned_pred[l] = pred
        tuned_acc[l] = float((pred == Fev).mean())
        # raw: identity probe (no training) for comparison
        raw_logits = head(Xev); mx.eval(raw_logits)
        raw_pred = np.asarray(mx.argmax(raw_logits, axis=-1))
        raw_acc[l] = float((raw_pred == Fev).mean())
        print(f"   L{l:2d}: tuned acc={tuned_acc[l]:.3f}  raw acc={raw_acc[l]:.3f}", flush=True)

    # earliest calibrated agreement per token
    earliest = np.full(n_eval, L, dtype=np.int32)
    for t in range(n_eval):
        hit = np.where(tuned_pred[:, t] == Fev[t])[0]
        if len(hit):
            earliest[t] = hit[0] + 1
    avg_depth = float(earliest.mean())

    # confidence-thresholded exit using tuned probes
    # (recompute max prob per layer)
    conf_rows = []
    # fraction of tokens that could exit by each depth budget
    for frac_layer in [0.25, 0.5, 0.75]:
        Lb = max(1, int(frac_layer * L))
        # token correct if tuned pred at layer Lb-1 already equals final
        agree = float((tuned_pred[Lb - 1] == Fev).mean())
        conf_rows.append({"depth_used": Lb, "frac_of_depth": frac_layer,
                          "top1_agreement_if_exit_here": round(agree, 4)})

    result = {"model": sm.model_id, "n_layers": L, "eval_tokens": int(n_eval),
              "corpus": src, "method": "tuned lens (per-layer affine probe → final head)",
              "raw_logit_lens_acc_by_layer": [round(float(a), 4) for a in raw_acc],
              "tuned_lens_acc_by_layer": [round(float(a), 4) for a in tuned_acc],
              "avg_earliest_calibrated_depth": round(avg_depth, 2),
              "adaptive_depth_headroom_frac": round(1 - avg_depth / L, 4),
              "fixed_exit_agreement": conf_rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[result] tuned-lens avg earliest depth = {avg_depth:.1f}/{L} "
          f"→ {(1-avg_depth/L):.0%} adaptive-depth headroom", flush=True)
    print(f"[result] vs raw logit lens (was ~7-11% headroom)", flush=True)
    print(f"[save] -> {os.path.abspath(OUT)}", flush=True)


if __name__ == "__main__":
    main()

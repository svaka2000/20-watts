#!/usr/bin/env python3
"""
20 Watts — Episode 1 hardening: REALIZABLE sparsity via a trained predictor.

Oracle top-k tells you the *headroom*. To actually SKIP a neuron's compute you
must know it will be small BEFORE you compute it. Deja Vu (Liu et al., ICML 2023)
showed a cheap predictor can do this. Here we reproduce that idea at small scale:

For representative layers we train a tiny LOW-RANK predictor
    score(x) = (x · A) · B,     A: H×r,  B: r×I,   r ≪ H
from the MLP input x to the set of neurons that will be active (top-k by |h|).
We report:
  - recall@k  : of the truly-active neurons, what fraction the predictor finds
  - vs random : k/I baseline
  - output fidelity : ||down(h·pred_mask) − down(h)|| vs the oracle's error

High recall + near-oracle fidelity ⇒ the 52% headroom is realizable, not a trick.

Run: python src/predictor.py            (full: 6 layers)
     python src/predictor.py --quick    (2 layers, fast)
"""
import os, sys, json, time, argparse
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
sys.path.insert(0, os.path.dirname(__file__))
from sparse_patch import SparseModel

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "predictor_results.json")

LONG_FALLBACK = (
    "The history of science is a history of learning what to ignore. " * 40 +
    "Energy is conserved, information is compressed, and attention is selective. " * 40 +
    "A neuron stays silent until it has something worth saying, and so should a machine. " * 40
)


def build_corpus(target_tokens=6000):
    from_err = None
    try:
        from datasets import load_dataset
        for repo in ["Salesforce/wikitext", "wikitext"]:
            try:
                ds = load_dataset(repo, "wikitext-2-raw-v1", split="train")
            except Exception as e:
                from_err = e
                continue
            buf, n = [], 0
            for r in ds:
                t = r["text"].strip()
                if len(t) < 40:
                    continue
                buf.append(t); n += len(t.split())
                if n > target_tokens * 1.5:
                    break
            return "\n".join(buf), repo + "/wikitext-2-raw-v1"
    except Exception as e:
        from_err = e
    print("[warn] wikitext unavailable, using fallback corpus:", from_err, flush=True)
    return LONG_FALLBACK, "builtin-fallback"


def collect(sm, target_layers, keep, n_tokens, seq_len=512):
    """Run the model over a corpus, capturing (x, top-k target mask) for target layers."""
    text, source = build_corpus(target_tokens=n_tokens + 1500)
    ids = sm.encode(text)
    store = {lid: {"x": [], "y": []} for lid in target_layers}
    tl = set(target_layers)
    I = sm.dims["I"]
    k = max(1, int(round(keep * I)))

    def hook(lid, x, h):
        if lid not in tl:
            return
        absh = mx.abs(h)
        thr = mx.sort(absh, axis=-1)[..., I - k:I - k + 1]
        ymask = (absh >= thr)                     # [B,T,I] bool  (the active set)
        xf = x.reshape(-1, x.shape[-1])
        yf = ymask.reshape(-1, I)
        mx.eval(xf, yf)
        store[lid]["x"].append(np.asarray(xf).astype(np.float16))
        store[lid]["y"].append(np.asarray(yf).astype(np.uint8))

    sm.hook = hook
    sm.keep = 1.0                                  # capture true activations (dense)
    pos = 0
    collected = 0
    while pos < len(ids) and collected < n_tokens:
        chunk = ids[pos:pos + seq_len]
        if len(chunk) < 8:
            break
        out = sm.model(mx.array([chunk])); mx.eval(out)
        pos += seq_len
        collected = sum(a.shape[0] for a in store[target_layers[0]]["x"])
    sm.hook = None
    data = {}
    for lid in target_layers:
        X = np.concatenate(store[lid]["x"], axis=0)[:n_tokens]
        Y = np.concatenate(store[lid]["y"], axis=0)[:n_tokens]
        data[lid] = (X, Y)
    return data, source, k


class LowRank(nn.Module):
    def __init__(self, H, r, I):
        super().__init__()
        self.A = nn.Linear(H, r, bias=False)
        self.B = nn.Linear(r, I, bias=False)

    def __call__(self, x):
        # bottleneck MLP (Deja Vu-style); ReLU adds expressivity at the same FLOP cost
        return self.B(nn.relu(self.A(x)))


def bce_logits(z, y):
    # softplus(z) - y*z  == log(1+e^z) - y z
    return mx.mean(mx.logaddexp(mx.zeros_like(z), z) - y * z)


def train_predictor(X, Y, H, I, r=512, steps=400, bs=256, lr=1e-3, n_eval=1000):
    Xtr, Ytr = X[:-n_eval], Y[:-n_eval]
    Xev, Yev = X[-n_eval:], Y[-n_eval:]
    model = LowRank(H, r, I)
    opt = optim.Adam(learning_rate=lr)

    def loss_fn(m, xb, yb):
        return bce_logits(m(xb), yb)
    lg = nn.value_and_grad(model, loss_fn)

    N = Xtr.shape[0]
    Xtr_m = mx.array(Xtr.astype(np.float32))
    Ytr_m = mx.array(Ytr.astype(np.float32))
    for step in range(steps):
        idx = mx.array(np.random.randint(0, N, size=bs))
        xb = Xtr_m[idx]; yb = Ytr_m[idx]
        loss, grads = lg(model, xb, yb)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)

    # eval recall@k
    k = min(I - 1, max(1, int(round(Yev[0].sum()))))   # clamp: mask can be all-true at high keep
    Xev_m = mx.array(Xev.astype(np.float32))
    logits = model(Xev_m); mx.eval(logits)
    logits = np.asarray(logits)
    recalls = []
    for i in range(Xev.shape[0]):
        pred_top = np.argpartition(-logits[i], k)[:k]
        true_mask = Yev[i].astype(bool)
        recalls.append(true_mask[pred_top].sum() / max(1, k))
    return model, float(np.mean(recalls)), k, (Xev_m, Yev)


def fidelity(sm, lid, Xev_m, model, keep):
    """Compare layer output under predicted vs oracle vs dense masking."""
    mlp = sm.layers[lid].mlp
    x = Xev_m[:256].astype(mx.float16)
    gate = mlp.gate_proj(x); up = mlp.up_proj(x)
    h = nn.silu(gate) * up
    I = h.shape[-1]; k = max(1, int(round(keep * I)))
    # oracle
    absh = mx.abs(h)
    thr = mx.sort(absh, axis=-1)[..., I - k:I - k + 1]
    h_or = h * (absh >= thr)
    # predicted
    sc = model(x)
    thr_p = mx.sort(sc, axis=-1)[..., I - k:I - k + 1]
    h_pr = h * (sc >= thr_p)
    # magnitude-weighted recall: fraction of total |h| mass kept by the predicted set
    # (this, not exact membership, is what determines output fidelity)
    total = mx.abs(h).sum(-1)
    mass_pred = (mx.abs(h) * (sc >= thr_p)).sum(-1) / (total + 1e-9)
    mass_recall = float(mx.mean(mass_pred))
    y0 = mlp.down_proj(h); yo = mlp.down_proj(h_or); yp = mlp.down_proj(h_pr)
    mx.eval(y0, yo, yp)
    y0 = np.asarray(y0).astype(np.float64)
    rel = lambda a: float(np.linalg.norm(np.asarray(a).astype(np.float64) - y0) / (np.linalg.norm(y0) + 1e-9))
    return rel(yo), rel(yp), mass_recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    print("[load] model ...", flush=True)
    sm = SparseModel(verify=True)
    H, I, L = sm.dims["H"], sm.dims["I"], sm.dims["L"]
    print(f"[load] diff={sm.max_diff:.0e} H={H} I={I} L={L}", flush=True)

    if args.quick:
        target_layers = [7, 20]; n_tokens = 2500; steps = 150
    else:
        target_layers = [2, 7, 13, 18, 23, 27]; n_tokens = 8000; steps = args.steps
    target_layers = [l for l in target_layers if l < L]

    print(f"[collect] layers={target_layers} keep={args.keep} ...", flush=True)
    t0 = time.time()
    data, source, k = collect(sm, target_layers, args.keep, n_tokens)
    print(f"[collect] {source}: {n_tokens} tok/layer in {time.time()-t0:.1f}s; k={k}", flush=True)

    rows = []
    for lid in target_layers:
        X, Y = data[lid]
        t1 = time.time()
        model, recall, kk, (Xev_m, Yev) = train_predictor(X, Y, H, I, r=args.rank, steps=steps)
        rel_or, rel_pr, mrec = fidelity(sm, lid, Xev_m, model, args.keep)
        rows.append({"layer": lid, "recall_at_k": round(recall, 4),
                     "random_baseline": round(args.keep, 4),
                     "pred_mass_recall": round(mrec, 4),
                     "oracle_output_relerr": round(rel_or, 4),
                     "predicted_output_relerr": round(rel_pr, 4)})
        print(f"   L{lid:2d}: recall@k={recall:.3f} mass={mrec:.3f}  "
              f"relErr oracle={rel_or:.3f} pred={rel_pr:.3f}  [{time.time()-t1:.1f}s]", flush=True)

    mean_recall = float(np.mean([r["recall_at_k"] for r in rows]))
    mean_mass = float(np.mean([r["pred_mass_recall"] for r in rows]))
    # predictor FLOP cost vs one projection, to be honest about overhead
    pred_flops = args.rank * (H + I)
    proj_flops = H * I
    result = {"model": sm.model_id, "keep": args.keep, "rank": args.rank,
              "corpus": source, "n_tokens_per_layer": n_tokens, "k_active": k,
              "mean_recall_at_k": round(mean_recall, 4),
              "mean_pred_mass_recall": round(mean_mass, 4),
              "random_baseline_recall": args.keep,
              "predictor_flops_per_token_per_layer": int(pred_flops),
              "one_projection_flops": int(proj_flops),
              "predictor_overhead_vs_projection": round(pred_flops / proj_flops, 3),
              "per_layer": rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[result] mean recall@k = {mean_recall:.3f}  (random {args.keep})", flush=True)
    print(f"[result] mean predicted MASS recall = {mean_mass:.3f} "
          f"(fraction of activation magnitude kept)", flush=True)
    print(f"[result] predictor overhead = {pred_flops/proj_flops:.2f}× one projection", flush=True)
    print(f"[save] -> {os.path.abspath(OUT)}", flush=True)


if __name__ == "__main__":
    main()

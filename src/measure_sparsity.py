#!/usr/bin/env python3
"""
20 Watts -- Episode 1: Sparse Firing
====================================
Measure the INTRINSIC activation sparsity of a real 7B LLM and the
quality/compute trade-off of exploiting it ("neuron skipping"), on
Apple Silicon via MLX.

This is an HONEST measurement. We carefully separate two quantities:

  (a) Intrinsic / oracle sparsity  -- how many MLP neurons can be skipped
      per token, at a given output quality. Measured exactly here.

  (b) Realizable compute savings   -- to actually SKIP a neuron's compute
      you need to know in advance which neurons will be small. That requires
      a cheap predictor (cf. Deja Vu, Liu et al., ICML 2023). We therefore
      report BOTH an upper bound (predictor skips gate+up+down) and a lower
      bound (skip only the down-projection), and never conflate "skippable"
      with "free."

Neuroscience anchor: in cortex, the metabolic cost of a spike limits the
fraction of neurons that can be active at once to < 1% (Lennie, 2003,
"The Cost of Cortical Computation"). A dense LLM "fires" 100% of its MLP
neurons on every token. This script asks: how lazy can we make it?

Run:
    python src/measure_sparsity.py            # full run
    python src/measure_sparsity.py --quick    # fast smoke test
Outputs:
    results/sparsity_results.json
"""
import os, sys, json, glob, time, math, argparse
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

MODEL = os.environ.get("WATTS_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
OUT   = os.path.join(os.path.dirname(__file__), "..", "results", "sparsity_results.json")

# ----------------------------------------------------------------------------
# Diverse probe prompts (to show sparsity is universal, not domain-specific)
# and a held-out passage for perplexity (original prose -- no license issues).
# ----------------------------------------------------------------------------
PROMPTS = [
    "The mitochondria is the powerhouse of the cell because",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Q: What is the capital of France?\nA:",
    "Once upon a time, in a village at the edge of a great forest,",
    "The integral of x squared with respect to x is equal to",
    "Dear hiring manager, I am writing to apply for the position of",
    "In 1969, the Apollo 11 mission successfully landed the first humans",
    "The stock market reacted sharply this morning after the central bank",
    "import numpy as np\narr = np.array([1, 2, 3])\nprint(arr.mean())  #",
    "She looked at him and whispered, \"I never meant for any of this to",
]

EVAL_TEXT = (
    "Energy is the currency of every living system. A leaf, when struck by "
    "sunlight, does not attempt to capture every photon with perfect fidelity. "
    "Instead it relies on a small number of pigment molecules tuned to the most "
    "useful wavelengths, and it discards the rest as heat. This selective "
    "strategy is not a flaw; it is the entire reason the process is efficient "
    "enough to power a forest. The same principle appears again and again across "
    "biology. A nerve cell does not fire continuously. It stays silent for long "
    "stretches and spends its precious metabolic budget only on the signals that "
    "matter, because the act of firing is expensive and the brain cannot afford "
    "to run every neuron at once. Researchers have estimated that fewer than one "
    "in a hundred cortical neurons can be substantially active at any single "
    "moment. From this constraint a surprising elegance emerges. Memory, "
    "perception, and thought are all built on top of sparseness, on the quiet "
    "majority of cells that wait while a chosen few carry the message forward. "
    "Engineers who design artificial networks have only recently begun to take "
    "this lesson seriously. A modern language model, by contrast, activates "
    "every one of its internal units for every word it reads, whether the word "
    "is surprising or utterly predictable. It pays the full price of computation "
    "even when almost no computation is required. If a machine could learn to be "
    "lazy in the way a brain is lazy, to stay silent unless a unit truly has "
    "something to contribute, then the cost of intelligence might fall not by a "
    "few percent but by a large factor, and the gap between a warehouse of "
    "graphics cards and a twenty watt organ might begin, at last, to close. "
    "The history of computing is in many ways a history of doing less. Early "
    "machines recomputed every value from scratch, until engineers realized that "
    "storing intermediate results could save enormous effort. Later, processors "
    "learned to predict which instructions a program would need next and to skip "
    "the work that would otherwise have been wasted. Every major leap in "
    "efficiency came from the same insight, that most of the work a naive system "
    "performs is unnecessary, and the art lies in identifying the small fraction "
    "that truly matters. A language model is, at its heart, an enormous "
    "arithmetic engine, and it has so far been built in the most naive way "
    "imaginable, multiplying every number by every weight for every word. The "
    "opportunity hiding inside it is the same one the brain exploited long ago "
    "and that computer architects rediscover in every generation. Consider the "
    "ocean, which appears uniform from a distance but is in truth mostly empty "
    "water punctuated by rare concentrations of life. A fish does not search "
    "every cubic meter for food; it follows gradients, attends to the few cues "
    "that signal a meal, and ignores the vast still expanse between them. "
    "Attention itself is a form of sparseness, a decision about where not to "
    "look. When a system learns to ignore, it does not become less capable. It "
    "becomes faster, cheaper, and often more accurate, because it is no longer "
    "distracted by the irrelevant. The promise of sparse computation is not "
    "merely to save power. It is to build machines that, like living things, "
    "spend their effort only where effort is deserved."
)


def find_config(model_id):
    name = "models--" + model_id.replace("/", "--")
    base = os.path.expanduser(f"~/.cache/huggingface/hub/{name}/snapshots")
    for cfg in glob.glob(os.path.join(base, "*", "config.json")):
        with open(cfg) as f:
            return json.load(f)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke test")
    args = ap.parse_args()

    print(f"[load] {MODEL} ...", flush=True)
    t0 = time.time()
    model, tokenizer = load(MODEL)
    print(f"[load] done in {time.time()-t0:.1f}s", flush=True)

    cfg = find_config(MODEL)
    H   = cfg["hidden_size"]
    I   = cfg["intermediate_size"]
    L   = cfg["num_hidden_layers"]
    nH  = cfg["num_attention_heads"]
    nKV = cfg.get("num_key_value_heads", nH)
    hd  = H // nH
    print(f"[cfg] H={H} I={I} L={L} heads={nH} kv={nKV} head_dim={hd}", flush=True)

    # ---- per-token FLOP accounting for the linear projections ----
    attn_flops = 2 * H * H + 2 * H * (nKV * hd)   # q,o + k,v (GQA-aware)
    mlp_flops  = 3 * H * I                          # gate, up, down
    mlp_frac   = mlp_flops / (attn_flops + mlp_flops)
    print(f"[flops] MLP share of linear-projection FLOPs = {mlp_frac:.3f}", flush=True)

    # ---- locate decoder layers + MLP class ----
    layers = model.model.layers
    mlp0   = layers[0].mlp
    MLPClass = type(mlp0)
    _orig_call = MLPClass.__call__
    for i, l in enumerate(layers):
        l.mlp.layer_id = i

    # shared mutable control state
    STATE = {"mode": "off", "keep": 1.0}
    COV_K   = [0.05, 0.10, 0.25, 0.50]   # report mass captured by top-k% neurons
    REL_TAU = [0.01, 0.05]               # near-zero relative to per-token max |h|
    acc = {}

    def _ensure(lid):
        if lid not in acc:
            acc[lid] = {"n": 0,
                        "cov": {k: 0.0 for k in COV_K},
                        "nz":  {t: 0.0 for t in REL_TAU}}
        return acc[lid]

    def record(lid, h):
        absh = mx.abs(h)
        mx.eval(absh)
        # cast to float64: summing ~19k fp16 values overflows (>65504) and
        # corrupts the cumulative-mass statistic. fp64 is exact here.
        a = np.asarray(absh).astype(np.float64).reshape(-1, h.shape[-1])   # [N, I]
        N = a.shape[0]
        s = _ensure(lid); s["n"] += N
        order = np.sort(a, axis=1)[:, ::-1]               # descending
        total = order.sum(axis=1) + 1e-9
        csum  = np.cumsum(order, axis=1)
        for k in COV_K:
            idx = max(1, int(round(k * a.shape[1]))) - 1
            s["cov"][k] += float((csum[:, idx] / total).sum())
        ptm = order[:, 0:1]
        for tau in REL_TAU:
            fb = (a < tau * ptm).mean(axis=1)             # [N] frac below tau*max
            s["nz"][tau] += float(fb.sum())

    def topk_mask(h, keep):
        Ii = h.shape[-1]
        k = max(1, int(round(keep * Ii)))
        absh = mx.abs(h)
        thr = mx.sort(absh, axis=-1)[..., Ii - k:Ii - k + 1]   # smallest kept mag
        return h * (absh >= thr)

    def patched(self, x):
        gate = self.gate_proj(x)
        up   = self.up_proj(x)
        h = nn.silu(gate) * up
        if STATE["mode"] == "stats":
            record(self.layer_id, h)
        if STATE["keep"] < 1.0:
            h = topk_mask(h, STATE["keep"])
        return self.down_proj(h)

    # ---- integrity check: patched == original at keep=1 (proves reimplement is exact) ----
    probe = mx.random.normal((1, 4, H)).astype(mx.float16)
    orig_out = _orig_call(mlp0, probe)
    MLPClass.__call__ = patched                # install patch
    STATE["mode"], STATE["keep"] = "off", 1.0
    new_out = mlp0(probe)
    mx.eval(orig_out, new_out)
    max_diff = float(mx.abs(orig_out - new_out).max())
    print(f"[sanity] max|patched-original| at keep=1.0 = {max_diff:.2e}", flush=True)
    assert max_diff < 1e-2, "Patched MLP does not match original -- aborting."

    def forward_logits(ids):
        return model(ids)

    def perplexity(ids):
        logits = forward_logits(ids)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tgt = ids[:, 1:]
        lp = mx.take_along_axis(logp[:, :-1, :], tgt[..., None], axis=-1)[..., 0]
        nll = -lp.mean()
        mx.eval(nll)
        nll = float(nll)
        return nll, math.exp(nll)

    # ========================= 1) STATS PASS =========================
    prompts = PROMPTS[:3] if args.quick else PROMPTS
    print(f"[stats] running {len(prompts)} probe prompts ...", flush=True)
    STATE["mode"], STATE["keep"] = "stats", 1.0
    for p in prompts:
        ids = mx.array([tokenizer.encode(p)])
        out = model(ids); mx.eval(out)
    STATE["mode"] = "off"

    # aggregate per-layer + overall
    per_layer = []
    overall_cov = {k: [] for k in COV_K}
    overall_nz  = {t: [] for t in REL_TAU}
    for lid in sorted(acc):
        s = acc[lid]; n = max(1, s["n"])
        cov = {k: s["cov"][k] / n for k in COV_K}
        nz  = {t: s["nz"][t] / n for t in REL_TAU}
        per_layer.append({"layer": lid, "tokens": s["n"], "topk_mass": cov, "frac_below_rel_tau": nz})
        for k in COV_K: overall_cov[k].append(cov[k])
        for t in REL_TAU: overall_nz[t].append(nz[t])
    overall = {
        "topk_mass": {f"top_{int(k*100)}pct": float(np.mean(v)) for k, v in overall_cov.items()},
        "frac_below_rel_tau": {f"tau_{t}": float(np.mean(v)) for t, v in overall_nz.items()},
    }
    print(f"[stats] overall top-k mass: {overall['topk_mass']}", flush=True)
    print(f"[stats] overall near-zero frac: {overall['frac_below_rel_tau']}", flush=True)

    # ========================= 2) MASK / PERPLEXITY SWEEP =========================
    eval_ids = mx.array([tokenizer.encode(EVAL_TEXT)])
    n_eval_tok = eval_ids.shape[1]
    keeps = [1.0, 0.5, 0.3, 0.2] if args.quick else \
        [1.0, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
    print(f"[ppl] held-out eval tokens = {n_eval_tok}; sweeping keep={keeps}", flush=True)
    STATE["mode"] = "off"
    sweep = []
    nll0 = None
    for keep in keeps:
        STATE["keep"] = keep
        nll, ppl = perplexity(eval_ids)
        if nll0 is None: nll0 = nll
        sweep.append({
            "keep": keep, "skip": round(1 - keep, 3),
            "ppl": ppl, "nll": nll,
            "ppl_increase_pct": 100.0 * (math.exp(nll - nll0) - 1.0),
        })
        print(f"   keep={keep:.2f} skip={1-keep:4.0%}  ppl={ppl:8.3f}  "
              f"(+{sweep[-1]['ppl_increase_pct']:.2f}%)", flush=True)
    STATE["keep"] = 1.0

    # ---- translate the 'free' sparsity level into compute reduction ----
    def reduction_at(max_ppl_increase_pct):
        best = 0.0
        for row in sweep:
            if row["ppl_increase_pct"] <= max_ppl_increase_pct:
                best = max(best, row["skip"])
        return {
            "max_ppl_increase_pct": max_ppl_increase_pct,
            "skippable_neuron_frac": best,
            "compute_reduction_predictor_upper": round(best * mlp_frac, 4),    # skip gate+up+down
            "compute_reduction_downproj_lower": round(best * mlp_frac / 3, 4), # skip down only
        }
    budgets = [1.0, 2.0, 5.0]
    realizable = [reduction_at(b) for b in budgets]
    for r in realizable:
        print(f"[reduce] <= {r['max_ppl_increase_pct']}% ppl: skip {r['skippable_neuron_frac']:.0%} "
              f"of neurons -> {r['compute_reduction_predictor_upper']:.0%} compute (predictor) / "
              f"{r['compute_reduction_downproj_lower']:.0%} (down-only)", flush=True)

    # ========================= 3) QUALITATIVE GENERATION DEMO =========================
    demos = {}
    try:
        def greedy(prompt, n=30, keep=1.0):
            STATE["keep"] = keep
            ids = tokenizer.encode(prompt)
            for _ in range(n):
                lg = model(mx.array([ids]))[0, -1]
                nxt = int(mx.argmax(lg))
                ids.append(nxt)
                if tokenizer.eos_token_id is not None and nxt == tokenizer.eos_token_id:
                    break
            STATE["keep"] = 1.0
            return tokenizer.decode(ids)
        dp = "Explain in one sentence why the human brain is so energy efficient."
        demos["prompt"] = dp
        demos["dense_keep_1.0"]  = greedy(dp, keep=1.0)
        demos["sparse_keep_0.5"] = greedy(dp, keep=0.5)
        demos["sparse_keep_0.3"] = greedy(dp, keep=0.3)
        print("[demo] dense :", demos["dense_keep_1.0"], flush=True)
        print("[demo] 50% sparse:", demos["sparse_keep_0.5"], flush=True)
    except Exception as e:
        demos["error"] = repr(e)
        print("[demo] skipped:", e, flush=True)

    # ========================= SAVE =========================
    result = {
        "model": MODEL,
        "hardware": "Apple M4 Pro (MLX, Metal GPU)",
        "config": {"hidden_size": H, "intermediate_size": I, "num_layers": L,
                   "num_heads": nH, "num_kv_heads": nKV, "head_dim": hd,
                   "inter_over_hidden": round(I / H, 3)},
        "flop_accounting": {
            "attn_proj_flops_per_token": int(attn_flops),
            "mlp_proj_flops_per_token": int(mlp_flops),
            "mlp_share_of_linear_flops": round(mlp_frac, 4),
            "note": "Per-token linear-projection FLOPs; attention score/context "
                    "FLOPs (~T) are excluded and matter only at long context.",
        },
        "sanity_max_abs_diff_keep1": max_diff,
        "intrinsic_sparsity": {"per_layer": per_layer, "overall": overall},
        "perplexity_sweep": sweep,
        "realizable_reduction": realizable,
        "generation_demo": demos,
        "eval_tokens": int(n_eval_tok),
        "n_probe_prompts": len(prompts),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[save] -> {os.path.abspath(OUT)}", flush=True)


if __name__ == "__main__":
    main()

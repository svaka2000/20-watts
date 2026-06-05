#!/usr/bin/env python3
"""Generate figures for Episodes 2 & 3 and the predictor, from results/*.json.
Tolerant: only plots what exists."""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
FIGD = os.path.join(RES, "figures")
os.makedirs(FIGD, exist_ok=True)
INK = "#0b132b"; ACCENT = "#5bc0be"; WARN = "#e07a5f"; GOLD = "#e0a458"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK, "figure.dpi": 130,
                     "savefig.bbox": "tight"})


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ---- Ep2: layer importance + depth drop ----
ld = load("layer_drop_results.json")
if ld:
    imp = ld["layer_importance"]
    xs = [r["layer"] for r in imp]
    ys = [r["ppl_increase_pct"] for r in imp]
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.bar(xs, ys, color=ACCENT, edgecolor=INK, lw=0.4)
    ax.set_xlabel("Layer index"); ax.set_ylabel("Perplexity increase if dropped (%)")
    ax.set_title("Not all depth is load-bearing (Qwen2.5-7B)", fontweight="bold")
    ax.grid(alpha=0.2, axis="y")
    fig.savefig(os.path.join(FIGD, "ep2_layer_importance.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4))
    for key, lab, col in [("drop_last_k", "drop last K layers", WARN),
                          ("drop_least_important_k", "drop K least-important", INK)]:
        rows = ld[key]
        ax.plot([r["compute_saved_frac"] * 100 for r in rows],
                [r["ppl_increase_pct"] for r in rows], "-o", color=col, label=lab, ms=6)
    ax.axhspan(-1, 5, color=ACCENT, alpha=0.12, label="≤5% quality cost")
    ax.set_xlabel("Transformer compute saved (%)"); ax.set_ylabel("Perplexity increase (%)")
    ax.set_title("Episode 2: how much depth can you skip?", fontweight="bold")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.2)
    fig.savefig(os.path.join(FIGD, "ep2_depth_drop.png")); plt.close(fig)
    print("wrote ep2 figures")

# ---- Ep3: KV eviction ----
kv = load("kv_eviction_results.json")
if kv:
    rows = kv["sweep"]
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.plot([r["mem_saved_frac_at_T"] * 100 for r in rows],
            [r["ppl_increase_pct"] for r in rows], "-o", color=INK, ms=7, mfc=GOLD, mec=INK)
    ax.axhspan(-1, 5, color=ACCENT, alpha=0.12, label="≤5% quality cost")
    for r in rows:
        ax.annotate(f"W={r['window']}", (r["mem_saved_frac_at_T"] * 100, r["ppl_increase_pct"]),
                    fontsize=8, xytext=(3, 4), textcoords="offset points")
    ax.set_xlabel(f"KV-cache memory saved (%) at {kv['eval_tokens']} tokens")
    ax.set_ylabel("Perplexity increase (%)")
    ax.set_title("Episode 3: keep the gist, drop the transcript", fontweight="bold")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.2)
    fig.savefig(os.path.join(FIGD, "ep3_kv_eviction.png")); plt.close(fig)
    print("wrote ep3 figure")

# ---- Ep1: predictor recall ----
pr = load("predictor_results.json")
if pr:
    rows = pr["per_layer"]
    xs = [f"L{r['layer']}" for r in rows]
    rec = [r["recall_at_k"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.bar(xs, rec, color=ACCENT, edgecolor=INK)
    ax.axhline(pr["random_baseline_recall"] * 100, color=WARN, ls="--",
               label=f"random baseline ({pr['random_baseline_recall']:.0%})")
    ax.axhline(pr["mean_recall_at_k"] * 100, color=INK, ls=":",
               label=f"mean recall ({pr['mean_recall_at_k']:.0%})")
    ax.set_ylabel("Recall@k of active neurons (%)")
    ax.set_title("A cheap predictor finds the neurons that matter", fontweight="bold")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.2, axis="y")
    fig.savefig(os.path.join(FIGD, "ep1_predictor_recall.png")); plt.close(fig)
    print("wrote predictor figure")

# ---- Synthesis: stacked levers ----
st = load("stack_results.json")
if st:
    rows = st["stages"]
    names = [r["stage"].replace("+ ", "+\n").replace(" (", "\n(") for r in rows]
    ppl = [r["ppl_increase_pct"] for r in rows]
    comp = [r["compute_reduction_pct"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    x = list(range(len(rows)))
    ax.bar(x, comp, color=ACCENT, edgecolor=INK, label="compute reduction")
    ax.set_ylabel("Compute reduction (%)")
    ax2 = ax.twinx()
    ax2.plot(x, ppl, "-o", color=WARN, lw=2, label="perplexity cost")
    ax2.set_ylabel("Perplexity increase (%)", color=WARN)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_title("Synthesis: stacking brain-inspired levers (on 4-bit)", fontweight="bold")
    for i, v in zip(x, comp):
        ax.text(i, v + 1, f"-{v:.0f}%", ha="center", fontsize=8, fontweight="bold")
    fig.savefig(os.path.join(FIGD, "synthesis_stack.png")); plt.close(fig)
    print("wrote synthesis figure")

# ---- Ep3 H2O: heavy hitters vs recency ----
h2o = load("kv_h2o_results.json")
if h2o:
    rows = h2o["comparison"]
    labels = ["recency only" if r["mode"] == "stream" else "H2O\n(+ heavy hitters)" for r in rows]
    vals = [r["ppl_increase_pct"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.bar(labels, vals, color=[WARN, ACCENT], edgecolor=INK, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.2, f"+{v:.0f}%", ha="center", fontweight="bold")
    ax.set_ylabel("Perplexity increase (%)")
    ax.set_title(f"Same {h2o['budget_tokens']}-token budget (92% evicted):\nheavy hitters win",
                 fontweight="bold", fontsize=12)
    ax.grid(alpha=0.2, axis="y")
    fig.savefig(os.path.join(FIGD, "ep3_h2o.png")); plt.close(fig)
    print("wrote h2o figure")

# ---- Ep4: static pruning vs dynamic firing ----
sp = load("synaptic_prune_results.json")
if sp:
    rows = sp["comparison"]
    skip = [r["skip"] * 100 for r in rows]
    dyn = [r["dyn_increase_pct"] for r in rows]
    stat = [r["stat_increase_pct"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.axhspan(-5, 5, color=ACCENT, alpha=0.10, label="≤5% quality cost")
    ax.plot(skip, stat, "-o", color=WARN, lw=2.4, ms=7, label="STATIC pruning (one fixed set)")
    ax.plot(skip, dyn, "-o", color=INK, lw=2.4, ms=7, mfc=ACCENT, mec=INK,
            label="DYNAMIC sparse firing (per token)")
    ax.set_xlabel("% of MLP neurons removed")
    ax.set_ylabel("Perplexity increase vs dense (%)")
    ax.set_title("Why the brain stays dynamic (Qwen2.5-7B)", fontweight="bold")
    ax.annotate("permanent pruning\nexplodes", xy=(60, 259), xytext=(34, 700),
                color=WARN, fontweight="bold", fontsize=9,
                arrowprops=dict(arrowstyle="->", color=WARN))
    ax.annotate("per-token stays flat\n(free to 60%)", xy=(60, 0.8), xytext=(8, 250),
                color=INK, fontweight="bold", fontsize=9,
                arrowprops=dict(arrowstyle="->", color=INK))
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.2)
    fig.savefig(os.path.join(FIGD, "ep4_static_vs_dynamic.png")); plt.close(fig)
    print("wrote ep4 figure")

print("done ->", os.path.abspath(FIGD))

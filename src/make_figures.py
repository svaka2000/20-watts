#!/usr/bin/env python3
"""Generate publication figures from results/sparsity_results.json."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES  = os.path.join(HERE, "..", "results", "sparsity_results.json")
FIGD = os.path.join(HERE, "..", "results", "figures")
os.makedirs(FIGD, exist_ok=True)

with open(RES) as f:
    R = json.load(f)

INK = "#0b132b"; ACCENT = "#5bc0be"; WARN = "#e07a5f"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "text.color": INK,
                     "figure.dpi": 130, "savefig.bbox": "tight"})

# ---- Fig 1: perplexity vs neurons skipped (the money figure) ----
sw = R["perplexity_sweep"]
skip = [r["skip"] * 100 for r in sw]
inc  = [r["ppl_increase_pct"] for r in sw]
fig, ax = plt.subplots(figsize=(7, 4.3))
ax.axhspan(-5, 1, color=ACCENT, alpha=0.12, label="≤ 1% quality cost ('free' zone)")
ax.axhline(0, color=INK, lw=0.8, ls="--")
ax.plot(skip, inc, "-o", color=INK, lw=2, mfc=ACCENT, mec=INK, ms=7)
ax.set_xlabel("% of MLP neurons skipped per token")
ax.set_ylabel("Perplexity increase vs dense (%)")
ax.set_title("How lazy can a 7B model be?  (Qwen2.5-7B, MLX, Apple M4 Pro)",
             fontweight="bold")
# annotate the free point
free = R["realizable_reduction"][0]["skippable_neuron_frac"] * 100
ax.axvline(free, color=WARN, lw=1.2, ls=":")
ax.annotate(f"{free:.0f}% skipped\nat ~0 quality cost", xy=(free, 0.5),
            xytext=(free - 30, 8), color=WARN, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=WARN))
ax.legend(loc="upper left", frameon=False)
ax.grid(alpha=0.25)
fig.savefig(os.path.join(FIGD, "fig1_quality_vs_sparsity.png"))
plt.close(fig)

# ---- Fig 2: sparsity by layer depth ----
pl = R["intrinsic_sparsity"]["per_layer"]
lids = [p["layer"] for p in pl]
top25 = [p["topk_mass"]["0.25"] * 100 if "0.25" in p["topk_mass"]
         else list(p["topk_mass"].values())[2] * 100 for p in pl]
nz = [p["frac_below_rel_tau"]["0.05"] * 100 if "0.05" in p["frac_below_rel_tau"]
      else list(p["frac_below_rel_tau"].values())[1] * 100 for p in pl]
fig, ax = plt.subplots(figsize=(7, 4.0))
ax.bar(lids, nz, color=ACCENT, edgecolor=INK, lw=0.4,
       label="% neurons below 5% of peak magnitude")
ax.plot(lids, top25, "-o", color=WARN, ms=4, lw=1.6,
        label="% of activation 'mass' held by top 25% of neurons")
ax.set_xlabel("Transformer layer (depth)")
ax.set_ylabel("Percent")
ax.set_title("Activation sparsity is present at every depth", fontweight="bold")
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.2, axis="y")
fig.savefig(os.path.join(FIGD, "fig2_sparsity_by_layer.png"))
plt.close(fig)

# ---- Fig 3: realizable compute reduction at quality budgets ----
rr = R["realizable_reduction"]
labels = [f"≤ {r['max_ppl_increase_pct']:.0f}%" for r in rr]
upper = [r["compute_reduction_predictor_upper"] * 100 for r in rr]
lower = [r["compute_reduction_downproj_lower"] * 100 for r in rr]
x = range(len(rr))
fig, ax = plt.subplots(figsize=(7, 4.0))
ax.bar([i - 0.2 for i in x], upper, width=0.4, color=INK,
       label="predictor-realizable (skip gate+up+down)")
ax.bar([i + 0.2 for i in x], lower, width=0.4, color=ACCENT, edgecolor=INK,
       label="conservative (skip down-proj only)")
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.set_xlabel("Quality budget (max perplexity increase)")
ax.set_ylabel("Inference compute reduction (%)")
ax.set_title("Honest energy headroom from sparse firing", fontweight="bold")
for i, v in zip(x, upper):
    ax.text(i - 0.2, v + 1, f"{v:.0f}%", ha="center", fontweight="bold", fontsize=9)
for i, v in zip(x, lower):
    ax.text(i + 0.2, v + 1, f"{v:.0f}%", ha="center", fontsize=8, color=INK)
ax.set_ylim(0, max(upper) * 1.18)
ax.legend(frameon=False, fontsize=9, loc="upper center",
          bbox_to_anchor=(0.5, -0.18), ncol=2)
ax.grid(alpha=0.2, axis="y")
fig.savefig(os.path.join(FIGD, "fig3_compute_reduction.png"))
plt.close(fig)

print("Saved figures to", os.path.abspath(FIGD))
for fn in sorted(os.listdir(FIGD)):
    print("  -", fn)

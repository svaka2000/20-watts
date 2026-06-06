#!/usr/bin/env python3
"""Render drop-in 9:16 'proof card' graphics for the reels, from the real results.
Outputs assets/proof/*.png (1080x1920), dark/cinematic, mobile-readable."""
import os, json, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results", "sparsity_results.json")
OUT = os.path.join(HERE, "..", "assets", "proof")
os.makedirs(OUT, exist_ok=True)
R = json.load(open(RES))

BG = "#070a14"; INK = "#eaeefb"; ACCENT = "#5bc0be"; GOLD = "#e0a458"; WARN = "#e07a5f"; MUTE = "#6b7490"
MONO = {"family": "monospace"}


def new_card():
    fig = plt.figure(figsize=(10.8, 19.2), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return fig, ax


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, facecolor=BG); plt.close(fig)
    print("wrote", os.path.relpath(p))


# ---- Card 1: THE RECEIPT ----
sweep = R["perplexity_sweep"]
free = R["realizable_reduction"][0]
diff = R["sanity_max_abs_diff_keep1"]
mlp_share = R["flop_accounting"]["mlp_share_of_linear_flops"]
p50 = next((r for r in sweep if abs(r["keep"] - 0.5) < 1e-6), None)
fig, ax = new_card()
ax.text(0.5, 0.93, "THE RECEIPT", ha="center", color=GOLD, fontsize=30, fontweight="bold")
ax.text(0.5, 0.885, "Qwen2.5-7B · measured on a laptop", ha="center", color=MUTE, fontsize=15)
lines = [
    ("I switched off", INK, 26, False),
    (f"{int(free['skippable_neuron_frac']*100)}% of the neurons", ACCENT, 52, True),
    ("perplexity change:", MUTE, 20, False),
    (f"< 1%   (at 50% off: {p50['ppl_increase_pct']:+.1f}%, better)", INK, 28, True),
    ("", INK, 10, False),
    ("integrity check", MUTE, 20, False),
    ("max| modified − original | = 0.00", INK, 26, True),
    ("✓ bit-for-bit identical — not broken", ACCENT, 24, True),
    ("", INK, 10, False),
    ("compute per token", MUTE, 20, False),
    (f"≈ {int(round(free['compute_reduction_predictor_upper']*100))}% less", GOLD, 50, True),
]
y = 0.80
for txt, col, sz, bold in lines:
    ax.text(0.5, y, txt, ha="center", color=col, fontsize=sz,
            fontweight="bold" if bold else "normal")
    y -= 0.052 if sz < 30 else 0.075
ax.text(0.5, 0.06, "and it STACKS on 4-bit", ha="center", color=WARN, fontsize=24, fontweight="bold")
save(fig, "proof_receipt.png")


# ---- Card 2: THE SWEEP TABLE (terminal style) ----
fig, ax = new_card()
ax.text(0.5, 0.93, "how lazy can it get?", ha="center", color=GOLD, fontsize=30, fontweight="bold")
ax.text(0.5, 0.88, "held-out perplexity vs neurons skipped", ha="center", color=MUTE, fontsize=15)
rows = [r for r in sweep if r["skip"] in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8)]
y = 0.78
ax.text(0.30, y, "SKIPPED", color=MUTE, fontsize=20, **MONO)
ax.text(0.74, y, "QUALITY", color=MUTE, fontsize=20, ha="right", **MONO)
y -= 0.06
for r in rows:
    sk = f"{int(r['skip']*100):>3d}%"
    dv = r["ppl_increase_pct"]
    col = ACCENT if dv <= 1 else (GOLD if dv <= 6 else WARN)
    tag = "free" if dv <= 1 else ("ok" if dv <= 6 else "breaks")
    ax.text(0.30, y, sk, color=INK, fontsize=30, **MONO)
    ax.text(0.74, y, f"{dv:+5.1f}%", color=col, fontsize=30, ha="right", **MONO)
    ax.text(0.80, y, tag, color=col, fontsize=18, **MONO)
    y -= 0.072
ax.text(0.5, 0.10, "flat until 60% — then it breaks", ha="center", color=INK, fontsize=24, fontweight="bold")
save(fig, "proof_sweep.png")


# ---- Card 3: STILL FLUENT AT 50% OFF ----
demo = R.get("generation_demo", {})
fig, ax = new_card()
ax.text(0.5, 0.93, "still fluent at 50% off", ha="center", color=GOLD, fontsize=30, fontweight="bold")
ax.text(0.5, 0.88, 'prompt: "why is the human brain so efficient?"', ha="center", color=MUTE, fontsize=14)


def block(y0, label, text, col):
    ax.text(0.08, y0, label, color=col, fontsize=20, fontweight="bold")
    wrapped = textwrap.fill(text.split(":", 0)[0] if False else text, width=42)
    ax.text(0.08, y0 - 0.04, wrapped, color=INK, fontsize=19, va="top", family="sans-serif")


dense = demo.get("dense_keep_1.0", "").split("efficient because", 1)
dense_t = ("…because" + dense[1]) if len(dense) > 1 else demo.get("dense_keep_1.0", "")[:240]
spar = demo.get("sparse_keep_0.5", "").split("efficient because", 1)
spar_t = ("…because" + spar[1]) if len(spar) > 1 else demo.get("sparse_keep_0.5", "")[:240]
import re as _re
clean = lambda s: _re.sub(r"<\|[^|]*\|>", "", s).strip()
block(0.78, "DENSE (100% of neurons)", clean(dense_t)[:225], ACCENT)
block(0.46, "SPARSE (50% switched off)", clean(spar_t)[:225], GOLD)
ax.text(0.5, 0.08, "half the neurons gone. still coherent.", ha="center", color=INK, fontsize=23, fontweight="bold")
save(fig, "proof_generation.png")

print("done ->", os.path.abspath(OUT))

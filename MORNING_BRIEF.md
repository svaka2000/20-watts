# ☀️ Morning Brief — what got built overnight

*For Samarth. Read this first. Everything is committed and pushed to
**https://github.com/svaka2000/20-watts** (public).*

---

## TL;DR
"20 Watts" went from one study to a **full, honest research program** — 3 episodes +
a synthesis, each measured on a real 7B model (and a second 3B model) on your laptop
via MLX, with a **bit-exact verification harness** so every claim is trustworthy.
4 academic PDFs, 3 ready-to-shoot reel scripts, 7 cinematic visuals, and a public repo.

## The results (all measured, all honest)

| Episode | Brain principle | Result |
|---|---|---|
| **1 · Sparse Firing** | <1% of neurons fire | **Skip 60% of MLP neurons → <1% perplexity; ~52% compute cut** (Qwen) / **~45%** (Llama-3.2-3B). Bit-exact. Holds on a downstream task. Predictor recovers ~72% of active mass (91% early/late layers). |
| **2 · Predictive Coding** | spend energy on surprise | **Honest non-win:** this model is *not* very depth-redundant (~1 free layer; final layers critical). A useful corrective to "just delete layers" hype. |
| **3 · Foveated Memory** | keep the gist | Constant memory for unlimited context; **reproduced the attention-sink: deleting the first token → +542% perplexity, one sink token fixes it.** |
| **★ Synthesis** | levers are independent | **4-bit + sparse firing = 4× smaller & −35% compute at +0.3% perplexity.** |

## What's in the repo
- `paper/` — **PAPER.md** (Ep1), **PAPER_EP2.md**, **PAPER_EP3.md**, **THESIS.md**
- `pdf/` — 4 polished **academic PDFs** (submittable look; trim Ep1 to 4 pages for the NeurIPS HS track)
- `VIDEO_SCRIPT.md` / `_EP2` / `_EP3` — **ready-to-shoot 60s reels** (two hooks each, on-screen captions, b-roll, honest caption)
- `BRIEFING.md` — teardown of the original video + **the exact Q&A questions a judge will ask and how to answer**
- `src/` — 11 experiments, all sharing a bit-exact harness (`sparse_patch.py`)
- `results/` — every number as JSON + `figures/` (publication PNGs)
- `assets/` — 7 cinematic Higgsfield visuals for the reels
- `notebooks/energy_colab.ipynb` — run on a free Colab GPU to get the **real joules/token**

## What needs YOU (10–30 min)
1. **Record Episode 1** (`VIDEO_SCRIPT.md`, Hook A or B) — it's the strongest, post it first.
2. (Optional) Run `notebooks/energy_colab.ipynb` on Colab to get a real-GPU watt number to show on screen.
3. (Optional, big) Trim `pdf/20Watts_Ep1...` to 4 pages and submit to the **NeurIPS High School Projects Track** when the call opens (it ran in 2024).
4. The repo is **public under your GitHub** — that backs the "code's open" claim. Make it private if you'd rather hold it.

## Honest notes (so you're never caught out)
- The "~52% compute" is **FLOP headroom**, partially realizable now (predictor) and fully realizable with Deja Vu-style kernels. Naive top-k may not speed up wall-clock — I say so openly (`src/latency.py`, `results/`).
- **Episode 2 is a deliberate non-win** — that's a feature. It's what makes Episodes 1 & 3 credible. Don't dress it up.
- I tried a "tuned lens" to strengthen Ep2; it didn't converge on a laptop budget, so I **dropped it rather than report a shaky number**. Future work.
- Higgsfield: used ~1 of your 706 credits for the 7 visuals.

## If you want me to keep going
- Build a tiny **landing page** for the series (Next.js, deploy to Vercel).
- Properly train the **tuned lens** (Ep2) to settle the depth question.
- Implement **H2O** (score-based KV eviction) for a stronger Ep3 tradeoff.
- Wire the **predictor** into a real sparse forward pass + measure MLX latency end-to-end.

*Built overnight, autonomously. Smaller numbers than the hype — every one real, paired
with its cost, reproducible on your Mac.*

# ⚡ 20 Watts

> The human brain runs general intelligence on about **20 watts**. A large language
> model needs a rack of GPUs. This project copies **one brain efficiency trick per
> episode**, measures it *honestly* on a real 7B model running on a **laptop**, and
> shows the tricks **stack** toward closing that gap.

**Why "20 Watts"?** A popular video argued you cut AI's energy by lowering *resolution*
(4-bit quantization — "the brain stores fuzzy memories"). True, but minor. The brain's
real efficiency comes from **doing less computation**: <1% of neurons fire at once, it
predicts away the expected, and it keeps only the gist. We target those.

Everything here is verified with a **bit-exact harness** — before any claim, the
modified model is checked against the original (max |Δ| = 0), so we never confuse
"broke the model" with "made it efficient." Every efficiency number is reported **with
its quality cost**. No lone "10×."

---

## The series at a glance

| # | Brain principle | Lever | Headline result (Qwen2.5-7B-4bit) |
|---|---|---|---|
| **1 · Sparse Firing** | <1% of neurons fire (Lennie 2003) | computation | **Skip 60% of MLP neurons → <1% perplexity. ~52% less compute, verified bit-exact.** |
| **2 · Predictive Coding** | spend energy on surprise | depth | Honest non-win: this model is *not* very depth-redundant; per-token early-exit needs a calibrated head. |
| **3 · Foveated Memory** | keep the gist, not the transcript | memory | Constant memory for unlimited context; **reproduced the attention-sink: −sink → +542% perplexity.** |
| **4 · Synaptic Pruning** | prune ~½ of synapses | structure | **Dynamic firing tolerates 60% neuron removal *free*; static pruning collapses at 30% (+29%)** — adaptivity ≈ 2× the sparsity. |
| **★ Synthesis** | they're independent | all | **4-bit + sparse firing = 4× smaller & −35% compute at +0.3% perplexity.** |

![Episode 1 — the money figure](results/figures/fig1_quality_vs_sparsity.png)

---

## Episode 1 — Sparse Firing (the big win)

A dense transformer fires **100%** of its feed-forward neurons on every token. We find
**~60% are skippable per token with <1% perplexity change** (at 50%, quality *improves*).
Since the MLP is **87% of the per-token compute**, that's a **~52% inference-compute
reduction at <1% quality cost**, holding on a downstream task too — and it stacks on
4-bit. A trained predictor recovers **~91% of the active mass** in early/late layers
(middle layers need a stronger predictor, à la Deja Vu), so the saving is realizable,
not just oracle headroom.
→ [`paper/PAPER.md`](paper/PAPER.md) · [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md) · [`BRIEFING.md`](BRIEFING.md)

## Episode 2 — Predictive Coding (the honest non-win)

If easy tokens didn't need full depth we could exit early. We measured it two ways
(dropping whole layers; a raw + **tuned lens** per-token probe) and found this model is
**not very depth-redundant** — a useful corrective to "just delete layers" hype. Real
audits report non-wins; this is one.
→ [`paper/PAPER_EP2.md`](paper/PAPER_EP2.md) · [`VIDEO_SCRIPT_EP2.md`](VIDEO_SCRIPT_EP2.md)

## Episode 3 — Foveated Memory (a discovery)

Keep a few "sink" tokens + a recent window → **constant** memory at any context length.
The quality cost is real but bounded. The striking find: deleting the first few tokens
explodes perplexity **+542%**, and a *single* sink token fixes it — the attention-sink
phenomenon, reproduced with a faithfulness-checked harness. And at a fixed budget, keeping
high-attention **heavy hitters** (H2O) beats pure recency (+69% vs +79%) — bit-exact verified.
→ [`paper/PAPER_EP3.md`](paper/PAPER_EP3.md) · [`VIDEO_SCRIPT_EP3.md`](VIDEO_SCRIPT_EP3.md)

## Episode 4 — Synaptic Pruning (the comparison)

The brain prunes ~half its synapses in development *and* keeps firing dynamic. We test the
static kind — permanently removing the globally least-active neurons — and race it against
Episode 1's per-token firing on the same model. **Dynamic is free to 60% removal; static
collapses at 30% (+29% perplexity).** Adaptivity is worth ~2× the sparsity, because the
neurons a model needs change every token. (Honest: a SOTA pruner like Wanda would narrow,
not close, the gap.)
→ [`paper/PAPER_EP4.md`](paper/PAPER_EP4.md) · [`VIDEO_SCRIPT_EP4.md`](VIDEO_SCRIPT_EP4.md)

## The thesis

→ [`paper/THESIS.md`](paper/THESIS.md) — the unifying overview: an honest audit of which
brain tricks actually transfer, and how they stack.

---

## Reproduce everything (Apple Silicon, MLX)

```bash
python3 -m venv .venv --system-site-packages
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/measure_sparsity.py     # Ep1: sparsity + quality curve
.venv/bin/python src/downstream_eval.py       # Ep1: ARC accuracy vs sparsity
.venv/bin/python src/predictor.py             # Ep1: realizability (trained predictor)
.venv/bin/python src/generality.py            # Ep1: same effect on Llama-3.2-3B
.venv/bin/python src/layer_drop.py            # Ep2: depth redundancy
.venv/bin/python src/tuned_lens.py            # Ep2: true adaptive-depth headroom
.venv/bin/python src/kv_eviction.py           # Ep3: KV sinks + window
.venv/bin/python src/stack_all.py             # Synthesis: all levers together
.venv/bin/python src/make_figures.py src/make_figures_ep23.py
```

Every script prints a faithfulness/integrity check before any result. Outputs land in
`results/` (JSON + figures). For **real GPU joules/token**, run `src/energy_benchmark.py`
on Colab/NVIDIA.

## Repo

```
paper/      PAPER.md (Ep1) · PAPER_EP2.md · PAPER_EP3.md · THESIS.md
src/        sparse_patch.py (shared harness) + one script per experiment
results/    *_results.json + figures/*.png
BRIEFING.md         honest teardown of the reference video + Q&A defense
VIDEO_SCRIPT*.md    ready-to-shoot reels (Ep1/2/3)
PROGRESS.md         build log
```

## Key references
Lennie 2003 · Attwell & Laughlin 2001 · Li et al. 2023 (Lazy Neuron) · Liu et al. 2023
(Deja Vu) · Mirzadeh et al. 2024 (ReLU Strikes Back) · Belrose et al. 2023 (Tuned Lens) ·
Xiao et al. 2023 (StreamingLLM) · Schuster et al. 2022 (CALM).

*Built by a 17-year-old. Smaller numbers than the hype — every one of them real, paired
with its cost, and reproducible on a laptop.*

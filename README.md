# ⚡ 20 Watts — Episode 1: Sparse Firing

> The human brain runs on **~20 watts** — not because it stores fuzzy memories, but
> because **<1% of its neurons fire at any moment.** A language model fires **100%** of
> its neurons on every word. This project makes a 7B model *lazy like a brain* — and
> measures exactly what that's worth.

**Headline result (measured on Qwen2.5-7B-Instruct-4bit, Apple M4 Pro, MLX):**

| | |
|---|---|
| 🧠 Skip **60%** of MLP neurons per token | **<1% perplexity change** (at 50%, quality *improves*) |
| ⚡ Inference compute | **~52% lower** at <1% quality cost (≈57% at <5%) |
| ✅ Integrity | skip-path is **bit-for-bit identical** to the real model (diff = 0) |
| ✖️ Stacks on quantization | tested model is **already 4-bit** — this saving is *on top* |

![The money figure](results/figures/fig1_quality_vs_sparsity.png)

---

## The 30-second version

A viral video argued you can cut AI energy by **lowering resolution** (16-bit → 4-bit
quantization) — framed as "the brain stores low-res memories." True, but that's not the
brain's main trick. The brain's main trick is **silence**: spikes are expensive, so it
keeps almost every neuron off (Lennie, 2003).

So we target a *different* lever — **computation, not storage**. On every token we skip
the feed-forward neurons that are barely active. On a real 7B model, **~60% of them can
be switched off with no measurable quality loss**, cutting roughly **half the inference
compute** — and because the test model is already 4-bit, the two tricks **multiply**.

This isn't a new algorithm; activation sparsity is known (Lazy Neuron, ICLR'23; Deja Vu,
ICML'23). The contribution is a **rigorous, honest, reproducible measurement** on a
laptop, framed by the right biology, with a guarantee it didn't silently break the model.

## Why this beats the reference video (honestly)
- **Different, deeper brain principle** — sparse firing (the actual reason for 20 W), not low-res storage.
- **Numbers that survive a Q&A** — every compute figure is paired with its quality cost; no "10× vs 52%" contradiction.
- **Verified** — `max|patched − original| = 0`. We prove we didn't break the model.
- **Generous, not combative** — his lever and ours *stack*. 4-bit × sparse on the same model.

See **[BRIEFING.md](BRIEFING.md)** for the full teardown + how to defend this in a Q&A.

---

## Repository

```
20-watts/
├── README.md                  ← you are here
├── paper/PAPER.md             ← the study (abstract → results → refs)
├── BRIEFING.md                ← honest teardown of the reference video + defense guide
├── VIDEO_SCRIPT.md            ← ready-to-shoot 60s reel (Ep.1)
├── src/
│   ├── measure_sparsity.py    ← the experiment (MLX, Apple Silicon)
│   ├── energy_benchmark.py    ← real joules/token on an NVIDIA GPU (Colab)
│   └── make_figures.py        ← regenerate the figures
├── results/
│   ├── sparsity_results.json  ← every number, machine-readable
│   └── figures/*.png
└── requirements.txt
```

## Reproduce it (Apple Silicon, ~2 minutes)

```bash
python3 -m venv .venv --system-site-packages
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/measure_sparsity.py      # → results/sparsity_results.json
.venv/bin/python src/make_figures.py          # → results/figures/*.png
```

The script prints a sanity line proving the neuron-skipping is mathematically identical
to the model at full density, then sweeps quality vs sparsity.

## Measure the *real* energy (NVIDIA GPU / Google Colab)

The MLX run proves the **headroom**; this proves the **joules**. On a CUDA box:

```bash
pip install torch transformers accelerate pynvml
python src/energy_benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --keep 0.5
```

It samples GPU power via NVML during a fixed decode, repeats with warmup, and reports
**joules per token (mean ± std)** dense vs sparse — the fair unit a single wattage
reading can't give you.

---

## The series — closing the gap to a 20-watt brain
1. **Ep.1 — Sparse Firing** *(this)* · brain keeps 99% of neurons quiet → skip idle MLP neurons. *(compute)*
2. **Ep.2 — Predictive Coding** · brain only pays for *surprise* → early-exit on easy tokens. *(depth)*
3. **Ep.3 — Foveated Memory** · you keep the gist, not the transcript → KV-cache eviction. *(memory)*

Three brain principles, three efficiency levers, and they all stack.

## Key references
Lennie 2003 (*Cost of Cortical Computation*) · Attwell & Laughlin 2001 · Li et al. 2023 (*Lazy Neuron*, ICLR) · Liu et al. 2023 (*Deja Vu*, ICML) · Mirzadeh et al. 2024 (*ReLU Strikes Back*, ICLR). Full list in [paper/PAPER.md](paper/PAPER.md).

*Built by a 17-year-old. Smaller numbers than the other guy — but every one of them is real.*

# 20 Watts — Strategy & Defense Briefing
*Read this before you publish or post anything. It is the difference between "cool video" and "research that survives a Q&A."*

---

## 1. What the other creator actually did (be precise, not dismissive)

His claim: *"reduced the energy cost of LLaMA by 10×… 16-bit → 4-bit… 'interfor' quantization… 87W → 42W, 52% less energy/token, quality +2.5%."*

Translated into real terms:

- **"interfor quantization" = INT4 quantization.** Storing each weight in 4 bits instead of 16. This is a **completely standard, off-the-shelf technique** — GPTQ, AWQ, bitsandbytes, GGUF "Q4". `llama.cpp` ships it by default; millions of people run Q4 models. It is good engineering, but it is **not novel research**, and a reviewer will know that within seconds.
- **The biomimicry is narration, not mechanism.** "The brain stores low-resolution memories" is a real idea, but he did not *use* any brain principle to derive INT4 — INT4 existed years before and has nothing to do with neuroscience. The brain story was attached afterward.

### The cracks a judge will push on (and where he breaks)
1. **"10×" vs his own "52%."** 87→42 W is a **1.97× / 52%** reduction. Not 10×. The hook contradicts the data. This single inconsistency is enough to lose a Q&A.
2. **"Energy per token" from a single wattage reading.** A bare "87W → 42W" is a *power* reading at one instant, not *energy per token*. Lower wattage can even mean **slower** tokens (more total energy). Power ≠ efficiency unless you divide by throughput.
3. **"Quality only upgraded by 2.5%."** Quantization almost always *degrades* quality slightly; "upgraded" is the wrong word, and "+2.5%" with no metric named (perplexity? accuracy? on what data?) is unfalsifiable.
4. **No baseline rigor.** One run, no repeats, no error bars, no held-out set.

**You don't beat him by trashing him.** You beat him by doing the thing he gestured at — *for real* — and being able to answer all four questions above about your own work.

---

## 2. Why our study is a genuine step up (and *separate*)

| | His Episode | **Our Episode 1** |
|---|---|---|
| Brain principle | Low-resolution storage | **Sparse firing** (<1% of cortical neurons active at once — Lennie 2003) |
| What it changes | **Storage** (bits per weight) | **Computation** (which neurons run at all) |
| Technique | INT4 quantization (standard) | Top-k activation sparsity / neuron-skipping, grounded in Lazy-Neuron (ICLR'23) & Deja Vu (ICML'23) |
| Verified? | "trust me" | **Bit-exact integrity check: our skip-path == the real model at keep=1.0, max diff = 0.0** |
| Quality claim | "+2.5%" (undefined) | Perplexity on a held-out passage, exact %, full curve |
| Energy claim | one wattage reading | per-token FLOP accounting + a real-GPU joules/token protocol |

**The strategic kill shot:** storage and computation are **independent, multiplicative** levers. Our test model is *already INT4* (his trick). We then skip ~half the neurons (our trick) **on top of it** with no quality loss. So our framing is not "he's wrong" — it's **"his 4-bit and my sparsity stack. 4-bit × 2× sparsity on the same model."** That is a more sophisticated, more generous, and more defensible story than "I beat him."

---

## 3. Your numbers (filled in from the real run — see `results/sparsity_results.json`)

Headline facts you are allowed to say, because we measured them on a real 7B model:

- "**About 60% of a 7B model's feed-forward neurons do essentially nothing on any given word** — you can switch them off with **less than 1% change in perplexity**, and at 50% off perplexity actually *improves* slightly (11.027 → 11.006)."
- "Push to ~65% off and quality drops under 3%; only past ~80% off does it really break."
- "In this model the feed-forward block is **~87% of the per-token math**, so skipping ~60% of those neurons is a **~52% compute reduction at <1% quality cost (≈57% at <5%)** — measured per token, not from a single wattage blip."
- "And this is *on top of* 4-bit quantization, because the model we tested is already 4-bit."

> Always quote the **pair**: *how much you skipped* **and** *what it cost in quality*. Never a lone number. That habit is exactly what the other video lacked.

---

## 4. Hard questions you must be able to answer

**Q: "Isn't 'skippable' different from 'actually faster'?"**
Yes — and say so first; it earns trust. To *skip* a neuron's compute you must know it's small *before* computing it, which needs a cheap predictor. That's exactly what **Deja Vu (ICML 2023)** built, and they got real 2× wall-clock speedups on OPT-175B. We report two honest bounds: an upper bound (a predictor skips the gate, up, and down projections) and a conservative lower bound (skip only the down projection, which needs no prediction). The truth is in between and depends on the predictor.

**Q: "Did you just break the model and not notice?"**
No — the script runs an integrity check: it compares our custom forward pass to the model's real forward pass at keep=1.0 and asserts the max difference is ~0. It is in the code and in the results JSON.

**Q: "Is this just MoE / does this only work on toy models?"**
It's measured on Qwen2.5-7B-Instruct, a real production model, at every layer. It's related to but distinct from Mixture-of-Experts (MoE routes whole expert blocks at train time; this is *post-hoc, per-neuron, per-token* sparsity in a dense model). The Lazy-Neuron paper shows sparsity *grows* with scale, so the effect is expected to be **stronger** on bigger models, not weaker.

**Q: "Why does perplexity sometimes *improve* when you prune?"**
Removing the smallest activations is a denoiser; the Lazy-Neuron paper reports the same calibration/robustness benefit. We don't overclaim it — we just report it.

**Q: "What's the catch / limitation?"** (Volunteer these — it makes you credible.)
- Per-token top-k needs a fast top-k or a learned predictor to pay off on real hardware; our MLX numbers prove the *headroom*, the GPU script measures the *realized* joules.
- We measured one model family and short contexts; attention sparsity (a separate lever) dominates at long contexts — that's **Episode 3**.
- Perplexity is a proxy; a full study would add a downstream task (e.g., MMLU subset).

---

## 5. The series ("20 Watts" — closing the gap to a 20-watt brain)

- **Ep. 1 — Sparse Firing** *(this one).* The brain keeps ~99% of neurons quiet. We make a 7B LLM quiet too. *(computation)*
- **Ep. 2 — Predictive Coding.** The brain only spends energy on *surprising* input (Rao & Ballard 1999). Map to **early-exit / adaptive depth**: easy, predictable tokens leave the network early; only hard tokens use full depth (cf. CALM, NeurIPS'22). *(depth)*
- **Ep. 3 — Foveated Memory.** You don't store a perfect transcript of every word; you keep the gist. Map to **KV-cache eviction / attention sinks** (StreamingLLM, H2O). *(memory)*

Three episodes, three distinct brain principles, three distinct efficiency levers — and they all *stack*. That's a research program, not a one-off.

---

## 6. Where you could actually publish this
- **NeurIPS High School Projects Track** — ran in 2024 with 330+ submissions; 4-page papers, must be independent student work. Watch neurips.cc each summer for the call.
- Workshops at NeurIPS/ICLR/ICML (efficient-ML, "hardware-aware" tracks) accept short papers.
- arXiv (cs.LG) for a citable preprint immediately.
- The honest framing ("we measure intrinsic sparsity and bound the realizable savings") is exactly the tone reviewers reward.

*Be the person whose numbers are smaller but bulletproof. That person wins the room.*

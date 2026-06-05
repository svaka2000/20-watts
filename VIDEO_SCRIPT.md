# 🎬 20 Watts — Episode 1: "I Made LLaMA Lazy"
*Format mirrors the reference video (~50–60s vertical reel). Two hook options; pick one, A/B test.*

---

## HOOK (option A — the diss-by-precision)
> "A high schooler just said he cut AI's energy use by lowering its *resolution*, like your brain stores low-res memories. He's not wrong — but that's not actually why your brain only uses **20 watts**."

## HOOK (option B — the curiosity gap)
> "Right now, **99% of the neurons in your brain are doing nothing.** That's not a flaw. That's the entire reason you run on 20 watts and a chatbot runs on a power plant. So I forced an AI to do the same thing."

---

## SCRIPT

**[0:00–0:08]** *(on-screen: brain scan, only a few dots lighting up)*
Your brain doesn't store low-res memories to save energy. It saves energy by staying **silent.** A neuron firing is metabolically expensive, so at any instant **fewer than 1% of your neurons are active.** *(cite on screen: Lennie, 2003)*

**[0:08–0:18]** *(on-screen: a dense grid where EVERY cell lights up on every word)*
An AI like LLaMA does the opposite. For **every single word**, it fires **100% of its neurons** — even for a word as obvious as "the." It pays the full energy bill even when there's basically no thinking to do.

**[0:18–0:30]** *(on-screen: terminal, the top-k mask, then the curve from fig1)*
So I made it lazy. I took a real 7-billion-parameter model and, on every token, switched off the neurons that were barely firing. Watch the quality curve: I can switch off **60% of the neurons** — more than half — and the model gets **less than 1% worse.** At 50% off, it actually gets a tiny bit *better.*

**[0:30–0:40]** *(on-screen: "52% less compute" + "✓ bit-for-bit verified")*
That's **~52% less computation per token** — and I verified the model is mathematically identical when I turn the skipping off, so I know I didn't just break it. **Bit for bit.**

**[0:40–0:50]** *(on-screen: "4-bit (his) × sparse (mine)" two dials turning)*
And here's the part that matters: the model I tested was **already 4-bit** — already using that resolution trick. My trick is a *different* lever: his changes how data is **stored**, mine changes how much the model **computes.** They **multiply.** You can do both.

**[0:50–0:58]** *(on-screen: "20 WATTS · Ep. 1 of ∞")*
This is episode 1 of **20 Watts** — I'm copying one trick from the brain per episode until AI is as efficient as the 20-watt thing in your skull. Episode 2: the brain only spends energy on **surprising** information. So I'll make the AI lazy about *thinking*, not just *firing.*

**[CTA]**
Follow to watch a 17-year-old close the gap between a warehouse of GPUs and a human brain — with receipts. Code's open. Pin the perplexity curve.

---

## ON-SCREEN TEXT / CAPTIONS (burn-ins)
- "99% of your neurons: OFF right now"
- "LLaMA: 100% ON, every word"
- "skip 60% of neurons → <1% worse"
- "≈52% less compute / token"
- "✓ bit-for-bit verified (diff = 0)"
- "and it STACKS on 4-bit"
- "20 WATTS — Ep.1: Sparse Firing"

## B-ROLL / PROOF SHOTS (build trust — this is what he didn't show)
1. The terminal line: `[sanity] max|patched-original| at keep=1.0 = 0.00e+00`
2. `results/figures/fig1_quality_vs_sparsity.png` (the flat-then-cliff curve)
3. Live: the model generating fluent text with 50% of neurons disabled
4. The two-dial graphic: "STORAGE (4-bit)" × "COMPUTE (sparse)"

## HONESTY GUARDRAILS (so you never get caught out — read BRIEFING.md)
- Say **"~52% less compute per token,"** not "10×." Pair every number with its quality cost.
- Say **"skippable / headroom,"** then note real speedup needs a predictor (Deja Vu, ICML'23).
- Don't claim you invented activation sparsity — you **measured and verified** it on real hardware and framed it with the right brain principle. That's the honest, winning pitch.

## CAPTION (for the post)
> Your brain runs on 20 watts because 99% of it is OFF right now. So I switched off 60% of a 7B model's neurons — <1% quality drop, ~52% less compute, verified bit-for-bit, *on top of* 4-bit. Ep.1 of 20 Watts. Code open. 🧠⚡ #AI #neuroscience #MachineLearning

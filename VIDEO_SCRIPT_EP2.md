# 🎬 20 Watts — Episode 2: "I Was Wrong (On Purpose)"
*~55s reel. The honest-science angle. This episode builds TRUST, which makes Ep1 & Ep3 land harder.*

---

## HOOK (option A — the contrarian)
> "Everyone online says AI models are 'way too deep' and you can just delete half the layers. I tested that on a real 7-billion-parameter model. They're wrong — and the way they're wrong is fascinating."

## HOOK (option B — the integrity flex)
> "This is the episode where my big idea *failed.* I'm showing it to you anyway — because that's the whole reason you can trust the episodes where it worked."

---

## SCRIPT

**[0:00–0:10]** *(on-screen: a brain, only lighting up for a "surprising" word in a sentence)*
Your brain barely reacts to expected words — "the," "of," "and." It saves its energy for the *surprising* stuff. That's called predictive coding. So in Episode 2, I tried to make an AI do the same thing: think *shallow* on easy words, *deep* only on hard ones.

**[0:10–0:22]** *(on-screen: a tower of layers, trying to exit early)*
The popular claim is that big models are bloated with useless layers — just chop them off. There's even a famous tool, the "logit lens," that makes it *look* like the model knows its answer really early. If that were true, I could skip a ton of compute.

**[0:22–0:38]** *(on-screen: red X's; "only 1 layer free"; "last layers = critical")*
So I dropped layers one by one and measured the damage on a real 7B model. The result? **Only about one layer** could be removed for free. The *final* layers — the ones people say to delete first — turned out to be the most important. The model genuinely *needs* its depth. My idea mostly didn't work.

**[0:38–0:50]** *(on-screen: "logit lens = biased"; building a "tuned lens")*
And that famous logit lens? It's **biased** — it makes early layers look smarter than they are. I had to train my own calibrated version to even measure the truth. The hype tool was lying.

**[0:50–0:58]** *(on-screen: "20 WATTS · Ep.2 — the honest one")*
Real research has experiments that fail. I'm showing you mine — because that's exactly why you can trust Episode 1, where switching off 60% of the neurons actually *worked.* Episode 3: I delete the AI's memory. Follow for the honest version of AI efficiency.

---

## ON-SCREEN CAPTIONS
- "brains save energy for SURPRISE"
- "claim: AI is too deep, just delete layers"
- "reality: only ~1 layer was free"
- "the LAST layers were the most important 🤯"
- "the famous 'logit lens' is biased"
- "real science has failures. here's mine."

## B-ROLL / PROOF
1. `results/figures/ep2_layer_importance.png` (skewed importance)
2. `results/figures/ep2_depth_drop.png` (curve)
3. Terminal: faithfulness check (empty drop-set == dense)

## HONESTY GUARDRAILS
- This episode's POWER is its honesty — do not dress the non-win up as a win.
- Credit ShortGPT/Gromov (layer pruning) and Belrose (tuned lens) and CALM (early exit).
- The takeaway is "depth doesn't give a free lunch *on this model*," not "layer pruning never works."

## CAPTION
> The episode where my idea failed — and why I'm posting it anyway. Everyone says AI models are "too deep, just delete layers." I tested it on a real 7B model: only ~1 layer was free, and the LAST layers mattered most. Honest AI efficiency, Ep.2 of 20 Watts. 🧠⚡ #AI #science #ML

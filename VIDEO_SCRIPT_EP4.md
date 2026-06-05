# 🎬 20 Watts — Episode 4: "I Deleted Half the AI's Brain"
*~55s reel. The static-vs-dynamic reveal is genuinely surprising and visual.*

---

## HOOK (option A)
> "As you grew up, your brain *deleted* about half of its own connections. It's called synaptic pruning. So I did the same thing to an AI — and discovered the single biggest mistake people make when they try to shrink these models."

## HOOK (option B)
> "I removed 30% of an AI's neurons two different ways. One way barely touched it. The other way nearly destroyed it. Same neurons, same count — completely different outcome. Here's why."

---

## SCRIPT

**[0:00–0:10]** *(on-screen: a developing brain, connections being trimmed away)*
Here's something wild: as a child grows, the brain *prunes itself* — it deletes roughly half of its synaptic connections, keeping only what it uses. Engineers do the same thing to AI models to make them smaller. It's called pruning. So I tried it on a real 7-billion-parameter model.

**[0:10–0:24]** *(on-screen: "STATIC pruning" — neurons deleted permanently; ppl climbing)*
First, the way most people do it: find the neurons that are *least active on average* and delete them permanently. I removed just 30% — and the model got **29% worse.** Half? It more than **doubled** its error. Permanent pruning, even of the "useless" neurons, wrecked it.

**[0:24–0:40]** *(on-screen: "DYNAMIC firing" — neurons chosen per word; ppl flat)*
Then I removed the *same amount* a different way — letting the model pick which neurons to skip **fresh for every single word**, like a brain does. I skipped **60%** of them — twice as many — and the quality... didn't move. It was free.

**[0:40–0:52]** *(on-screen: the two curves diverging — flat vs exploding)*
Same number of neurons gone. One method explodes, the other is free. Why? Because the neurons a model actually needs **change with every word.** No single "deleted" set can be right for every input. That's the whole reason your brain prunes once — but stays *dynamic* forever.

**[0:52–0:58]** *(on-screen: "20 WATTS · Ep.4")*
Episode 4 of **20 Watts.** Storage, computation, depth, memory, and now structure — five brain tricks, measured honestly. Code's open. Follow for the finale.

---

## ON-SCREEN CAPTIONS
- "your brain deleted half its connections (synaptic pruning)"
- "delete 30% permanently → +29% worse"
- "skip 60% per-word → FREE"
- "same neurons. opposite result."
- "what you need changes every word"
- "prune once, stay dynamic forever"

## B-ROLL / PROOF
1. `results/figures/ep4_static_vs_dynamic.png` (the two curves diverging — the money shot)
2. Terminal: the static vs dynamic ppl table
3. `assets/ep4_synaptic_pruning.png`

## HONESTY GUARDRAILS
- Say our static metric is *activation-aware* (decent), and note SOTA pruners (Wanda/SparseGPT) do better — but can't beat the fundamental input-dependence point.
- Credit Lottery Ticket (Frankle), SparseGPT (Frantar), Wanda (Sun), and the neuroscience (Huttenlocher).
- The claim is "dynamic beats a fixed set because needs change per token," not "pruning is useless."

## CAPTION
> Your brain deleted half its connections growing up (synaptic pruning). I did it to a 7B AI: delete 30% permanently → +29% worse. Skip 60% per-word instead → free. Same neurons, opposite result — because what you need changes every word. Ep.4 of 20 Watts. 🧠⚡ #AI #neuroscience #ML

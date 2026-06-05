# 🎬 20 Watts — Episode 3: "I Deleted the AI's Memory"
*~55s vertical reel. The hook is the attention-sink discovery — it's genuinely surprising.*

---

## HOOK (option A)
> "I deleted **93% of an AI's memory** and it barely flinched. Then I deleted **one single word** at the start — and it completely fell apart. Here's the bizarre thing your brain and an AI both secretly do."

## HOOK (option B)
> "Your brain doesn't remember this sentence word for word — it keeps the gist. So I forced an AI to forget like a brain, and I found something nobody would guess."

---

## SCRIPT

**[0:00–0:10]** *(on-screen: a wall of text, most of it fading to grey, only the recent part bright)*
An AI remembers every word you've ever said to it — perfectly — in something called the KV cache. That's why long chats get slow and expensive. Your brain refuses to do this. It keeps the gist and the most recent details, and lets the rest blur.

**[0:10–0:22]** *(on-screen: slider cutting the memory down; "93% memory saved")*
So I made a 7-billion-parameter model forget like a brain: keep only the most recent words plus a few at the very start. I cut **93% of its memory** — and it still read fluently, just a bit worse. Cut less, lose almost nothing. Memory becomes **constant** no matter how long the conversation gets.

**[0:22–0:38]** *(on-screen: big number "+542%" flashing red, then a single token lights up)*
But here's the part that shocked me. When I deleted just the **first few words** of the context — words like the very first "The" — the model didn't get a little worse. Its error **exploded by 542%.** It broke. Add back **one** token and it's instantly fine again.

**[0:38–0:48]** *(on-screen: "attention sink" label on the first token)*
It turns out a trained AI secretly dumps its spare attention onto the first token, like a pressure-release valve. Researchers call it an **attention sink.** I reproduced it from scratch — and verified my code is mathematically identical to the real model first, so I know it's real, not a bug.

**[0:48–0:56]** *(on-screen: "20 WATTS · Ep.3")*
Episode 3 of **20 Watts**: storage, computation, depth, and now memory — four brain tricks, stacking toward a model as efficient as the 20-watt brain in your head. Code's open. The receipts are in the repo.

---

## ON-SCREEN CAPTIONS
- "AI: remembers every word. You: keep the gist."
- "cut 93% of memory → still fluent"
- "delete the FIRST token → +542% error 🤯"
- "add back ONE token → fixed"
- "it's called an attention sink"
- "✓ bit-for-bit verified harness"

## B-ROLL / PROOF
1. Terminal: `[check] causal ppl=5.914; streaming(W=inf) ppl=5.914; |Δ|=0.00` (faithfulness)
2. The ablation table: S=0 → +542%, S=1 → +73%
3. `results/figures/ep3_kv_eviction.png`

## HONESTY GUARDRAILS
- Don't say "free." Say: "constant memory for unlimited context, at a real but bounded quality cost."
- Credit StreamingLLM (Xiao et al., 2023) for the sink discovery — you *reproduced* it, which is the honest and still-impressive claim.

## CAPTION
> I deleted 93% of an AI's memory and it was fine. Then I deleted ONE token and it broke (+542% error). Meet the "attention sink" — the pressure valve every LLM secretly relies on. Ep.3 of 20 Watts. 🧠⚡ Code open. #AI #neuroscience #ML

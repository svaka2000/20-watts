# 🎬 20 Watts — Reel Edit Guide (paint-by-numbers)

Every line of every reel mapped to the exact file to put on screen. If you can read a
table and drag clips, you can cut these. **Record Episode 1 first.**

---

## Setup (5 minutes, once)
- **App:** CapCut (free, phone or desktop). New project → canvas **9:16 (1080×1920)**.
- **Import these folders:**
  - `assets/video/` — the b-roll clips (.mp4)
  - `assets/proof/` — the 3 "receipt" cards (.png)
  - `results/figures/` — the data figures (.png)
- **Record your voice** reading the script (Voice Memos is fine), drop it on the audio track, then lay visuals on top per the tables below.
- **Captions:** bold sans‑serif (Montserrat ExtraBold / CapCut "Proxima Nova"), **white, big, upper third**, subtle drop shadow. Burn in the **on‑screen text** column exactly.
- **Motion:** on still images (figures/proof cards) add a slow 5% zoom (Ken Burns) so nothing feels static. Cuts every **1.5–3s**.
- **Music:** one tense/cinematic bed from CapCut, volume ~15%, under your voice.
- **Cold‑open rule:** every reel opens on **`series_20watts.mp4`** for ~1.5s — the Virality Predictor rated it highest (58 vs ~49), and it's the series' face.
- **End every reel** with the CTA card: *"Follow — a 17-year-old vs the energy cost of AI. Code's open."* + your handle.

---

## ▶ REEL 1 — Episode 1: "I Made LLaMA Lazy"  *(post this first)*
★ **Hook to read (option B):** *"Right now, 99% of the neurons in your brain are doing nothing. That's the entire reason you run on 20 watts and a chatbot runs on a power plant. So I forced an AI to do the same thing."*

| t | SAY (voice) | SHOW (file) | ON‑SCREEN TEXT |
|---|---|---|---|
| 0:00–0:05 | *(the hook above)* | `assets/video/series_20watts.mp4` | **99% of your neurons: OFF right now** |
| 0:05–0:13 | "Your brain saves energy by staying **silent** — a neuron firing is expensive, so under **1%** are active at once." | `assets/video/ep1_brain_sparse.mp4` | fewer than 1% active · *Lennie, 2003* |
| 0:13–0:22 | "An AI does the **opposite**. For every single word it fires **100%** of its neurons — even for a word like 'the.'" | `assets/video/ep1_ai_dense.mp4` | **LLaMA: 100% ON, every word** |
| 0:22–0:31 | "So I made it lazy. On a real **7‑billion**‑param model I switched off the neurons barely firing. Watch the curve — **60% off, under 1% worse.** At 50% it gets *better.*" | `results/figures/fig1_quality_vs_sparsity.png` (hold) → `assets/proof/proof_sweep.png` | **skip 60% of neurons → <1% worse** |
| 0:31–0:41 | "That's **~52% less compute per token** — and I **verified** the model is mathematically identical when I turn skipping off. Bit for bit. I didn't break it." | `assets/proof/proof_receipt.png` | ✓ bit‑for‑bit (diff = 0) · ≈52% less compute |
| 0:41–0:50 | "And the model I tested was **already 4‑bit** — already using the 'resolution' trick. Mine's a **different** lever: storage vs compute. **They multiply.**" | `results/figures/synthesis_stack.png` *(or a text card: "4‑bit × sparse")* | **and it STACKS on 4‑bit** |
| 0:50–0:58 | "Episode 1 of **20 Watts** — one brain trick per episode until AI is as cheap as the thing in your skull. Next: the brain only pays for **surprise.**" | `assets/video/series_trailer.mp4` (tail) | **20 WATTS — Ep.1: Sparse Firing** |
| CTA | "Follow for a high‑schooler vs the energy cost of AI — with receipts. Code's open." | `assets/series_hero_thumbnail.png` | Follow · github.com/svaka2000/20‑watts |

*Optional B-roll under 0:22–0:41:* `assets/proof/proof_generation.png` ("still fluent at 50% off").
**Caption:** *Your brain runs on 20 watts because 99% of it is OFF right now. So I switched off 60% of a 7B model's neurons — <1% quality drop, ~52% less compute, verified bit‑for‑bit, on top of 4‑bit. Ep.1 of 20 Watts. Code open. 🧠⚡ #AI #neuroscience #MachineLearning*

---

## ▶ REEL 2 — Episode 3: "I Deleted the AI's Memory"  *(post 2nd — strongest shock)*
★ **Hook:** *"I deleted 93% of an AI's memory and it barely flinched. Then I deleted one single word at the start — and it completely fell apart."*

| t | SAY | SHOW (file) | ON‑SCREEN TEXT |
|---|---|---|---|
| 0:00–0:05 | *(hook)* | `assets/video/series_20watts.mp4` → `assets/video/ep3_memory_fade.mp4` | **deleted 93% of its memory…** |
| 0:05–0:14 | "An AI remembers every word you've said — perfectly — in its KV cache. Your brain refuses; it keeps the **gist** and the recent stuff." | `assets/video/ep3_memory_fade.mp4` | AI: every word. You: the gist. |
| 0:14–0:24 | "So I made a 7B model forget like a brain — keep a few early tokens + the recent window. **Cut 93% of its memory** and it still read fine." | `results/figures/ep3_kv_eviction.png` | **cut 93% of memory → still fluent** |
| 0:24–0:38 | "But here's the shock: delete just the **first few words** and the error didn't creep up — it **exploded +542%.** Add back **one** token and it's instantly fine." | `assets/video/ep3_attention_sink.mp4` | **delete the FIRST token → +542% 🤯** |
| 0:38–0:48 | "A trained AI secretly dumps spare attention onto the first token — an **attention sink.** I reproduced it from scratch, and verified my code is identical to the real model first." | `results/figures/ep3_h2o.png` | it's called an **attention sink** |
| 0:48–0:56 | "Episode 3 of 20 Watts. Storage, computation, depth, now **memory.** Code's open." | `assets/video/series_trailer.mp4` | **20 WATTS — Ep.3** |

**Caption:** *I deleted 93% of an AI's memory and it was fine. Then I deleted ONE token and it broke (+542% error). Meet the "attention sink" — the pressure valve every LLM secretly relies on. Ep.3 of 20 Watts. 🧠⚡ #AI #neuroscience #ML*

---

## ▶ REEL 3 — Episode 4: "I Deleted Half the Brain"  *(post 3rd)*
★ **Hook:** *"I removed 30% of an AI's neurons two different ways. One barely touched it. The other nearly destroyed it. Same neurons — completely different result."*

| t | SAY | SHOW (file) | ON‑SCREEN TEXT |
|---|---|---|---|
| 0:00–0:05 | *(hook)* | `assets/video/series_20watts.mp4` → `assets/video/ep4_synaptic_pruning.mp4` | same neurons. opposite result. |
| 0:05–0:14 | "As you grew up, your brain **deleted about half** of its own connections — synaptic pruning. Engineers prune AI the same way. So I tried it." | `assets/video/ep4_synaptic_pruning.mp4` | your brain deleted half its connections |
| 0:14–0:26 | "First the normal way: find the neurons least active **on average** and delete them **permanently.** I removed 30% — it got **29% worse.** Half? It **doubled** its error." | `results/figures/ep4_static_vs_dynamic.png` (the red curve) | **delete 30% permanently → +29% worse** |
| 0:26–0:40 | "Then the **brain's** way — let the model pick which to skip **fresh for every word.** I skipped **60%** — twice as many — and quality didn't move. **Free.**" | `assets/video/ep1_brain_sparse.mp4` → `results/figures/ep4_static_vs_dynamic.png` | **skip 60% per‑word → FREE** |
| 0:40–0:50 | "Same neurons gone — one explodes, one's free. Because the neurons a model needs **change every word.** That's why the brain prunes once but stays **dynamic** forever." | `results/figures/ep4_static_vs_dynamic.png` (both curves) | what you need changes every word |
| 0:50–0:57 | "Episode 4 of 20 Watts. Code's open." | `assets/video/series_trailer.mp4` | **20 WATTS — Ep.4** |

**Caption:** *Your brain deleted half its connections growing up. I did it to a 7B AI: delete 30% permanently → +29% worse. Skip 60% per‑word instead → free. Same neurons, opposite result. Ep.4 of 20 Watts. 🧠⚡ #AI #neuroscience #ML*

---

## ▶ REEL 4 — Episode 2: "I Was Wrong (On Purpose)"  *(post last — the trust builder)*
★ **Hook:** *"This is the episode where my big idea failed. I'm posting it anyway — because that's the whole reason you can trust the ones that worked."*

| t | SAY | SHOW (file) | ON‑SCREEN TEXT |
|---|---|---|---|
| 0:00–0:05 | *(hook)* | `assets/video/series_20watts.mp4` → `assets/video/ep2_depth.mp4` | real science has failures. here's mine. |
| 0:05–0:14 | "Your brain barely reacts to expected words and saves energy for **surprise** — predictive coding. So I tried to make an AI think **shallow** on easy words." | `assets/video/ep2_depth.mp4` | brains save energy for SURPRISE |
| 0:14–0:26 | "Everyone says big models are **too deep — just delete layers.** There's even a famous tool that makes it *look* like the model knows its answer super early." | `assets/video/ep2_depth.mp4` | claim: AI is too deep, just delete layers |
| 0:26–0:40 | "So I dropped layers and measured. **Only about one** could be removed for free — and the **final** layers, the ones people say to cut, were the **most important.**" | `results/figures/ep2_layer_importance.png` → `results/figures/ep2_depth_drop.png` | reality: only ~1 layer was free 🤯 |
| 0:40–0:50 | "And that famous tool? It's **biased** — makes early layers look smarter than they are. I had to train my own to even measure the truth." | `assets/video/ep2_depth.mp4` | the famous 'logit lens' is biased |
| 0:50–0:58 | "Real research has failures. I'm showing you mine — that's exactly why you can trust Episode 1, where it actually worked." | `assets/video/series_trailer.mp4` | **20 WATTS — Ep.2 (the honest one)** |

**Caption:** *The episode where my idea failed — and why I'm posting it anyway. Everyone says AI is "too deep, just delete layers." I tested it: only ~1 layer was free, and the LAST layers mattered most. Honest AI efficiency, Ep.2 of 20 Watts. 🧠⚡ #AI #science #ML*

---

## Asset cheat‑sheet (what each file shows)
**B‑roll video — `assets/video/`**
- `series_20watts.mp4` — lightbulb vs server rack *(use as every cold open — top virality)*
- `ep1_brain_sparse.mp4` — brain, a few neurons firing *(sparse firing)*
- `ep1_ai_dense.mp4` — grid, every node blazing *(the wasteful contrast)*
- `ep2_depth.mp4` — layered tower, light exiting early *(depth)*
- `ep3_attention_sink.mp4` — bright anchor pulling light *(attention sink)*
- `ep3_memory_fade.mp4` — text ribbon dissolving *(foveated memory)*
- `ep4_synaptic_pruning.mp4` — neuron with synapses trimming *(pruning)*
- `series_trailer.mp4` — 28s montage of all 7 *(outros / standalone teaser)*

**Proof cards — `assets/proof/`** (drop‑in, no terminal needed)
- `proof_receipt.png` — 60% off · diff=0 · ≈52% less · stacks on 4‑bit
- `proof_sweep.png` — skip% → quality table (flat then breaks)
- `proof_generation.png` — dense vs 50%‑sparse output, both fluent

**Figures — `results/figures/`** — `fig1_quality_vs_sparsity` (the hockey stick), `fig3_compute_reduction`, `ep1_predictor_recall`, `ep2_layer_importance`, `ep2_depth_drop`, `ep3_kv_eviction`, `ep3_h2o`, `ep4_static_vs_dynamic`, `synthesis_stack`.

## Posting plan
1. **Ep.1** (strongest, cleanest win) → 2. **Ep.3** (the +542% shock) → 3. **Ep.4** (the twist) → 4. **Ep.2** (the honest non‑win — frame as "story time, why you can trust me").
- Space them 2–4 days apart; reply to comments with the live link.
- Pin the comment: *"full method + code: github.com/svaka2000/20‑watts"*.
- Every script's full text, two hook options, and honesty guardrails live in `VIDEO_SCRIPT*.md`. Before you post a number, skim `BRIEFING.md` §4 (the Q&A defenses).

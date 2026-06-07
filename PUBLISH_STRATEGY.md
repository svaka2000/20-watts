# 20 Watts → Publication, Awards & College Playbook
*Honest, current (June 2026), and built for action. Sources linked at the bottom.*

---

## 0. The honest assessment (read this first)

**What you have:** a rigorous, bit-exact, reproducible **comparative study** of brain-inspired
efficiency levers in a real 7B model, plus a polished viral content series. The methodology
(integrity checks, paired numbers, an honest self-corrected negative result) is genuinely
above the typical high-school bar.

**The one weakness to fix for *top* awards:** most of the individual results **reproduce
known papers** (Lazy Neuron, Deja Vu, StreamingLLM, H2O, ShortGPT). Reproduction is real and
valuable, but the top competitions (Regeneron STS, Davidson, ISEF top awards) reward an
**original contribution**. The good news: you already have the seed of one —
**the static-vs-dynamic comparison ("adaptivity is worth ~2× the sparsity")** is a genuine,
defensible empirical finding. Sharpen *that* into the headline and you're competition-grade.

**You need two framings of the same work:**

| | Content framing (reels) | Academic framing (papers/awards) |
|---|---|---|
| Angle | "I made AI lazy — 52% less compute, beat this guy" | "Which brain-inspired efficiency principles transfer to LLMs, and *why*?" |
| Goal | reach / virality | publication / awards |
| Hypothesis | *(none — it's a flex)* | **"Dynamic, input-dependent sparsity preserves quality far better than static structural pruning at equal sparsity, because the set of important neurons is input-dependent."** |
| Venues reject the other framing? | n/a | **YES** — JEI/STS reject "my system works/outperforms." Use the hypothesis. |

> **Academic title to use:** *"Conditional vs. Structural Sparsity in a 7B Language Model:
> A Controlled Comparison of Brain-Inspired Efficiency Mechanisms."* The model is your
> **instrument**; the **scientific question** is when and why conditional computation beats
> structural compression. That framing is publishable everywhere below.

---

## 1. The venue map (what's real, what fits, what it costs)

| Venue | What you get | Fit | Effort | Timing (2026–27) |
|---|---|---|---|---|
| **arXiv preprint** | A citable, permanent, "I'm a published researcher" link. Legitimizes everything else. | ★★★★ | Low (paper's ~done) | Anytime — **needs an endorser** (see §3) |
| **Journal of Emerging Investigators (JEI)** | Real **peer-reviewed** publication built for high-schoolers. | ★★★★ (if reframed hypothesis-driven) | Low–med | Submit anytime; $35; **an adult must submit it** |
| **NeurIPS High School Track** | A *top-brand* line ("NeurIPS") that's actually accessible to solo HS work. Theme = ML for **social impact** → your **energy** angle fits perfectly. | ★★★★★ | Med (4-page paper) | **Watch neurips.cc late summer 2026** (ran 2024; conf is Dec 2026) |
| **Regeneron Science Talent Search (STS)** | The most prestigious US HS science award. Finalists → national press, $25k–$250k. | ★★★ (needs novelty + ideally a mentor) | High (20-page paper, essays, recs) | **Due Nov 5, 2026, 8pm ET — seniors only** |
| **Regeneron ISEF** | International fair; huge prestige; big scholarships. | ★★★ | High (fair circuit + forms) | **Qualify via a local/regional affiliated fair (winter 2026–27)** |
| **Davidson Fellows** | $25k / $50k / $100k scholarship for a "significant" body of work. | ★★★ (needs depth) | High | **2027 app opens Fall 2026** (US citizen/PR only) |
| **ML workshop paper** (NeurIPS/ICLR efficient-ML, "tiny papers") | Highest academic credibility per the field. | ★★ (high bar; needs novelty) | High | Per-workshop CFPs through the year |
| **JSHS** (DoD-sponsored) | Regional→national, scholarships. | — | — | **Currently suspended — check jshs.org** |
| **Show HN + X/TikTok launch** | Reach, GitHub stars, possible press — the "go big" engine. | ★★★★★ | Low | **This week** |

---

## 2. The 3-phase roadmap

### Phase 0 — Legitimize + launch (this week, ~6 hrs)
1. **Write the academic version** of the paper (I can do this): one ~8-page PDF titled as in §0,
   leading with the static-vs-dynamic hypothesis and the comparative audit. Keep the reel
   framing *out* of it.
2. **Get an arXiv endorser** and post the preprint (§3). Now you have a citable paper.
3. **Submit to JEI** — reframed hypothesis-driven, with an **adult** (teacher/parent/mentor) as
   the submitter. 70–75% of submissions get published; this is a near-lock credential.
4. **Launch the content:** post Reel 1 (open with the 20-watts clip — your Virality Predictor's
   pick), then a **Show HN** and an **X thread** (drafts in §4). Pin the repo link.

### Phase 1 — Sharpen to award-grade (next 4–8 weeks)
Pick **one** novelty to deepen — recommended: **the "value of adaptivity" study.** Make it bullet-proof:
- Add a **real static baseline** (Wanda: weight×activation) so static pruning isn't a strawman.
- Run it across **≥3 model families/sizes** (Qwen, Llama-3.2, + Mistral/Phi) → generality.
- Add a **mechanism**: measure how much the *active-neuron set overlaps across tokens*. Low
  overlap = high adaptivity value = a real explanation, not just a number.
- Keep the honest **end-to-end realizability** negative (trivial predictor compounds to +93%).
- **Stretch (genuine novelty):** design a *better* cheap predictor that actually realizes some
  of the headroom end-to-end. If it works, that's an original method — STS/ISEF gold.

### Phase 2 — Run the award circuit (Fall 2026 →)
- **STS** (if you're a senior): the 20-page paper is mostly your sharpened study. Due **Nov 5**.
- **NeurIPS HS track:** submit the 4-page version when the call opens (~late summer).
- **ISEF:** register for your **local Society-affiliated fair** now-ish; win up the chain.
- **Davidson Fellows:** apply in the 2027 cycle (~Feb 2027).

---

## 3. The keystone move: get a mentor (do this first)

One person unlocks almost everything: **arXiv endorsement**, the **JEI adult-submitter**, an
**STS recommendation**, an **ISEF sponsor**, and the credibility judges look for. A mentor is
the single highest-leverage thing you can add.

**Who to ask (in order):**
1. A **PhD student or professor in an ML lab** at a nearby university (UCSD is right there — its
   ML/NLP groups are ideal). PhD students reply more than professors.
2. **An author you cite** (Lazy Neuron / Deja Vu / StreamingLLM teams). They sometimes mentor or
   at least endorse motivated students — and any of them can be your **arXiv endorser** (find the
   endorser link at the bottom of their paper's arXiv page).
3. A **CS teacher** or a family/community contact with a research background (enough to submit JEI).
4. A paid structured program if you want a guaranteed mentor + workshop target (Polygence,
   Veritas AI, Algoverse) — optional, costs money.

**Cold-email template (copy/paste, fill the brackets):**
> Subject: HS student — feedback on a sparsity study (reproduced your [PAPER] on Qwen2.5-7B)
>
> Dr. [Name] — I'm a 17-year-old who reproduced [their finding] bit-for-bit on Qwen2.5-7B and
> ran a controlled comparison of *dynamic* vs *static* sparsity (code + data:
> github.com/svaka2000/20-watts). I found dynamic tolerates ~2× the neuron removal of static
> pruning at equal quality, and I'm trying to turn it into a proper paper. Would you be open to
> 15 minutes of feedback, or — if it's reasonable — an arXiv endorsement? Either way, thank you
> for [PAPER]; it's why I started.

Keep it short, specific, and lead with the fact you already did the work. The repo is your proof.

---

## 4. The "go big" launch kit

**Show HN (news.ycombinator.com/submit):**
> **Show HN: 20 Watts — I reproduced 5 LLM-efficiency papers bit-for-bit on a laptop (and where they break)**
> A 17-year-old's open, bit-exact audit of brain-inspired LLM efficiency on Qwen2.5-7B via MLX:
> skip 60% of MLP neurons for <1% perplexity (oracle), the attention-sink (+542% if you drop one
> token), and a controlled finding that *dynamic* sparsity beats *static* pruning ~2×. I also
> stress-tested my own realizability claim and it failed end-to-end (+93%) — included honestly.
> Code, papers, figures: github.com/svaka2000/20-watts

**X thread opener:**
> I'm 17. I spent a week forcing a 7B AI to run like a brain — switching off 60% of its neurons
> for <1% quality loss, on my laptop. Then I tried to break my own claim and it broke. 🧵 a
> brutally honest thread on what actually makes AI efficient ↓

(Then 6–8 tweets = the 4 episodes + the honest negative + the repo. Reuse the reel captions.)

**Where to post:** X, TikTok/Reels/Shorts (the 4 reels), Hacker News, r/MachineLearning
("[P]" project tag), LinkedIn. Reply to every comment; that's where reach compounds.

---

## 5. The college narrative (how this becomes admissions gold)

Top schools reward a **spike** — a deep, authentic, *demonstrated* obsession — over being
well-rounded. This project is a spike: **"ML-efficiency researcher + science communicator who
values honesty over hype."** You can prove it with artifacts most applicants can't:
a **paper** (arXiv/JEI), an **open-source repo** (stars = third-party validation), a **viral
series** (reach numbers), and a **live site**.

- **Common App Activities:** lead with this. *"Independent ML research on brain-inspired model
  efficiency; published [JEI/arXiv]; built a 4-part explainer series (X views); open-source repo."*
- **Additional Information / portfolio link:** drop `github.com/svaka2000/20-watts`.
- **Essays:** your *strongest* essay is the **honesty** beat — you made a claim, tested it, it
  failed, and you published the failure. That's intellectual character, and almost no applicant
  has it. (The reframed Ep2 "I was wrong on purpose" is literally an admissions essay.)
- **What it signals:** initiative, real research ability, communication, and integrity — the four
  things the most selective schools actually screen for.

**Honest caveat:** this is powerful *supporting* evidence, not a guarantee. It works because it's
real and you can discuss it deeply (that's what `BRIEFING.md` prepared you for). Don't inflate it
in apps — the restraint is part of the signal.

---

## 6. Do-this-now checklist
- [ ] Tell me your **graduation year**, **citizenship/PR status**, and whether you have any
  **mentor/teacher** access — it changes the priority order (see questions I'll ask).
- [ ] I write the **academic-framed paper** (hypothesis-driven, §0 title).
- [ ] Email **3 ML PhD students / cited authors** (template §3) for feedback + an arXiv endorser.
- [ ] **Submit to JEI** (reframed) with an adult submitter.
- [ ] **Post Reel 1 + Show HN + X thread** (kit in §4 / EDIT_GUIDE.md).
- [ ] Register for your **local ISEF-affiliated fair**; calendar **STS Nov 5** if you're a senior.

---

## Sources
- Regeneron STS 2027 (due Nov 5 2026; seniors; independent research w/ data; ≤20pp): [societyforscience.org/regeneron-sts/application-requirements](https://www.societyforscience.org/regeneron-sts/application-requirements/)
- Regeneron ISEF (grades 9–12; qualify via affiliated fair): [societyforscience.org/isef](https://www.societyforscience.org/isef/) · [international rules](https://www.societyforscience.org/isef/international-rules/)
- NeurIPS High School Projects (2024 call; theme = social impact; 4 pages): [neurips.cc/Conferences/2024/CallforHighSchoolProjects](https://neurips.cc/Conferences/2024/CallforHighSchoolProjects) · [2024 results](https://blog.neurips.cc/2024/11/18/announcing-the-neurips-high-school-projects-results/)
- Davidson Fellows ($25k–$100k; 18 or under; US citizen/PR; 2027 opens Fall 2026): [davidsongifted.org/fellows-scholarship/eligibility](https://www.davidsongifted.org/gifted-programs/fellows-scholarship/eligibility/)
- JEI submission + ML rules (hypothesis-driven; adult submits; $35): [emerginginvestigators.org/submissions/guidelines](https://emerginginvestigators.org/submissions/guidelines) · [ML/engineering rules](https://emerginginvestigators.org/submissions/engineering-and-machine-learning-based-projects)
- arXiv endorsement (no affiliation → personal endorser): [info.arxiv.org/help/endorsement.html](https://info.arxiv.org/help/endorsement.html)
- JSHS (grades 9–12; currently suspended — verify): [jshs.org](https://www.jshs.org/)

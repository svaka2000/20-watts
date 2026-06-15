# Outreach kit — get the mentor / endorser (the keystone move)

One aligned person unlocks the **arXiv endorsement**, the **JEI adult-submitter**, an **STS
recommendation**, and credibility. Your unfair advantage: **you already did the work** — lead
with the repo, not a request. Send 4–6 of these; you need one "yes."

**Rules:** email **PhD students before professors** (they reply far more). Keep it <120 words.
Subject line states the result. Attach nothing — link the repo. Follow up once after 5–7 days.

---

## 🎯 Target list (in priority order)

1. **Hao AI Lab @ UCSD** — `hao-ai-lab.github.io` (Prof. Hao Zhang). They work on **efficient LLM
   inference/serving** — your exact topic, and it's local (San Diego). **Email 2–3 of their PhD
   students** (find addresses on the lab "People" page / their personal sites). *Top pick.*
2. **UCSD NLP / ML-Systems group** — e.g. Prof. Taylor Berg-Kirkpatrick (NLP). Broader, still aligned.
3. **Authors of the papers you cite** (remote, great for an arXiv endorsement + feedback):
   - *Deja Vu* (contextual sparsity) — Beidi Chen (CMU) and her students.
   - *Wanda* (pruning) — Mingjie Sun, Zico Kolter (CMU).
   - *The Lazy Neuron Phenomenon* — Zonglin Li et al.
   Find current emails on their lab pages / arXiv author pages.
4. **A research-literate adult you already know** (a CS teacher, a family friend with a PhD) — only
   needs to be the **JEI submitter** (low effort; see Email C).

---

## ✉️ Email A — local PhD student at the Hao AI Lab (your best shot)

> **Subject:** HS student — dynamic vs static sparsity on Qwen2.5-7B (your lab's area)
>
> Hi [First name],
>
> I'm a 17-year-old in San Diego who's been working on LLM inference efficiency — your lab's
> area, which is why I'm writing you specifically.
>
> I reproduced contextual/activation sparsity bit-for-bit on Qwen2.5-7B (via MLX, on a laptop)
> and ran a controlled comparison: at equal sparsity, **dynamic per-token neuron selection
> tolerates ~2× the removal of the best static pruning** — and I measured *why*: the best fixed
> 50% set captures only **61%** of any given token's active neurons (Jaccard 0.39 across tokens).
> I also stress-tested realizability and it fails end-to-end (+93%), which I report honestly.
>
> Code, data, paper: **github.com/svaka2000/20-watts** (`paper/ACADEMIC_PAPER.md`).
>
> Could I get **15 minutes** of feedback on whether this is worth turning into a workshop paper —
> and, if it's reasonable, an **arXiv endorsement (cs.LG)**? Either way, thank you — your group's
> work is part of why I went down this path.
>
> [Your name] · [phone] · [school]

## ✉️ Email B — a cited author (remote feedback + endorsement)

> **Subject:** Reproduced your [PAPER] on Qwen2.5-7B — one question
>
> Dr. [Last name] — your paper **[PAPER]** is the reason I started this. I'm a 17-year-old who
> reproduced it bit-for-bit on Qwen2.5-7B and extended it into a controlled dynamic-vs-static
> sparsity comparison (dynamic tolerates ~2× the removal; I measured the input-dependence that
> explains it). Repo + paper: github.com/svaka2000/20-watts.
>
> Would you be open to a quick sanity-check on the framing, or — if appropriate — an **arXiv
> cs.LG endorsement**? I know your time is scarce; even a one-line "yes/no, and here's the
> obvious flaw" would mean a lot. Thank you for [PAPER].
>
> [Your name]

## ✉️ Email C — the JEI adult-submitter (teacher / parent / mentor)

> **Subject:** Quick favor — submitting my research paper (you'd be listed as the adult sponsor)
>
> Hi [Name] — I finished an independent ML research paper and want to submit it to the **Journal
> of Emerging Investigators**, a peer-reviewed journal for student researchers. Their one rule is
> that an **adult** (not the student) clicks "submit" and is listed as a sponsor — it's a 10-minute
> form, $35, and they publish ~75% of submissions. I'd do all the writing and revisions; I just
> need you as the submitter. Paper + repo: github.com/svaka2000/20-watts. Could you help?
>
> [Your name]

---

## 🔑 How the arXiv endorsement works
arXiv needs a one-time **endorsement** in cs.LG for a first-time author with no academic email.
- Easiest: **a mentor/PhD student who already publishes on arXiv** endorses you (one click).
- Or: open any cited paper's arXiv abstract page → there's a list of who can endorse for that
  category; email one with Email B. Once endorsed, you can post the preprint yourself.
- Then your paper has a **permanent, citable arxiv.org link** — put it everywhere.

## ⏱️ Follow-up (send 5–7 days later if no reply)
> Hi [First name] — quick bump on the below. Totally understand if you're swamped; even a pointer
> to a student or a "not for me" is helpful. Repo's here if useful: github.com/svaka2000/20-watts. Thanks!

---

*Want me to turn these into Gmail drafts (I have your Gmail wired) once you tell me the exact
recipients? I can also tailor Email B per cited author.*

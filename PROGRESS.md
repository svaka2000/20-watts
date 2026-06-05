# 20 Watts — Overnight Build Tracker
*Autonomous run started 2026-06-05. Each continuation updates this file and commits.*

**Thesis:** copy one brain efficiency trick per episode, measure it honestly on a real 7B
model, and show they STACK toward a 20-watt brain. Beat the reference video on rigor, not hype.

**Hardware/env:** Apple M4 Pro, MLX, `mlx-community/Qwen2.5-7B-Instruct-4bit`, venv at `.venv`.

---

## STATUS: Phase 1 — hardening Episode 1

### Phase 0 — Infra ✅
- [x] caffeinate 12h
- [x] shared instrumentation `src/sparse_patch.py`
- [x] git repo + GitHub push
- [x] memory: overnight-nonstop rule

### Phase 1 — Make Episode 1 bulletproof
- [ ] **Downstream accuracy** (`src/downstream_eval.py`) — ARC-Easy MCQ accuracy vs sparsity (answers the "perplexity is a proxy" critique)
- [ ] **Trained predictor** (`src/predictor.py`) — low-rank predictor of active neurons per layer; report recall@k and predicted-mask perplexity vs oracle → proves savings are REALIZABLE (Deja Vu-style), not just oracle headroom
- [ ] **Real latency** (`src/latency.py`) — wall-clock tokens/sec dense vs sparse on MLX; honest about top-k overhead
- [ ] **Generality** — repeat core sparsity measurement on a 2nd model family (Llama-3.2-3B-Instruct-4bit) to show it's not Qwen-specific

### Phase 2 — Episode 2: Predictive Coding (early-exit / adaptive depth)
- [ ] `src/early_exit.py` — per-token layer at which the prediction stabilizes; confidence-thresholded early exit; compute saved vs quality
- [ ] paper/PAPER_EP2.md + figures + video script

### Phase 3 — Episode 3: Foveated Memory (KV-cache eviction)
- [ ] `src/kv_eviction.py` — attention-sink + recent-window eviction (StreamingLLM-style); long-context perplexity + memory saved
- [ ] paper/PAPER_EP3.md + figures + video script

### Phase 4 — The Grand Synthesis
- [ ] `src/stack_all.py` — 4-bit × sparse firing × early exit × KV eviction on ONE model; total compute/energy reduction with quality curve
- [ ] paper/THESIS.md — the unifying "20 Watts" overview paper

### Phase 5 — Publish & polish
- [ ] Higgsfield cinematic visuals for the reels (brain/neuron/AI)
- [ ] NeurIPS-HS-formatted PDF of Episode 1
- [ ] series landing README + clean commit history

---

## RESULTS LOG (running)
- **Ep1 core:** skip 60% MLP neurons → <1% ppl change (50%→ better); MLP=87% FLOPs → ~52% compute cut @<1% quality, 57% @<5%; integrity diff=0; stacks on 4-bit.
- **Ep1 downstream:** ARC-style MCQ accuracy holds (0.875 dense = 0.875 at 40% skip) — confirms ppl story on a real task.
- **Ep1 predictor:** low-rank predictor of active neurons; smoke recall 0.68 vs 0.50 random, predicted-mask err≈oracle. Full run (rank 1024) in progress.
- **Ep2 layer-drop:** HONEST/nuanced — only ~1 layer freely droppable; last layers critical; layer 8 removal *improves* ppl 2.5%. Anti-hype vs "just delete layers."
- **Ep2 logit-lens:** raw lens says predictions stabilize late (≈11% headroom) — known lens artifact → building tuned lens for true number.
- **Ep3 KV eviction:** faithfulness |Δppl|=0; keep 26% of cache (W=512) = +34% ppl; **attention-sink discovery reproduced: S=0→+542%, S=1→+73%.**
- **Synthesis (stack_all):** robust free stack = 4-bit + sparse firing (skip 40%) → −35% compute @ +0.3% ppl + 4× storage; depth/KV add more at real cost (+6%/+68%).
- **Ep1 downstream (full ARC-150):** acc 0.747 dense → 0.740 / 0.727 / 0.720 / 0.700 / 0.687 at skip 40/50/60/70/80% — graceful.
- **Ep1 generality (Llama-3.2-3B):** bit-exact (diff 0); skip 60% <1% ppl → ~45% compute (MLP 75%). Not Qwen-specific.
- **Ep1 latency (honest):** naive top-k 479 → 389 tok/s (0.81×, slower) — sort overhead; motivates the predictor/fused kernel.
- **Predictor (final):** mean recall@k 0.66, mean mass-recall 0.72 (0.91 early/late, 0.60 middle).
- **Tuned lens:** did not converge on laptop budget → dropped (honesty over a shaky stat).
- **Deliverables:** 4 academic PDFs, landing page (GitHub Pages: svaka2000.github.io/20-watts), 7 Higgsfield visuals, Colab energy notebook, 3 reel scripts, BRIEFING, MORNING_BRIEF.

- **Ep3 H2O upgrade:** built a bit-exact manual-attention harness (manual ppl = fused = 6.048, |Δ|=0); at equal 132-tok budget, heavy-hitters (H2O) beat pure recency **+69% vs +79% ppl** — a real win reproduced from scratch.

## STATUS: CORE COMPLETE + Ep3 H2O upgrade — series shipped, public, reproducible. Optional extensions in MORNING_BRIEF.

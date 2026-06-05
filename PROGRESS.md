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
- **Ep1 core (done earlier):** skip 60% MLP neurons → <1% ppl change (50%→ better); MLP=87% FLOPs → ~52% compute cut @<1% quality, 57% @<5%; integrity diff=0; stacks on 4-bit.

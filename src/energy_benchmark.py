#!/usr/bin/env python3
"""
20 Watts -- Episode 1: Sparse Firing  (GPU ENERGY companion)
============================================================
The MLX study (src/measure_sparsity.py) measures how many MLP neurons are
*skippable* on Apple Silicon. This script measures the thing a skeptic actually
cares about: **joules per token** on a real NVIDIA GPU, dense vs. sparse.

Why a separate script? Honest energy numbers require:
  1. A real power sensor sampled at high frequency (pynvml -> the same counter
     `nvidia-smi` reads), integrated over a FIXED decode workload.
  2. Warmup + repeats + a quiet GPU, so you measure the model and not the noise.
  3. Reporting the trade-off (quality vs energy), never a single hero number.

Unlike a one-off "87W -> 42W" reading, this reports mean +/- std over repeats,
energy PER TOKEN (the only fair unit), and the quality cost at that operating
point. Run it on Google Colab (free T4) or any CUDA box.

Install:
    pip install torch transformers accelerate pynvml
Run:
    python energy_benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --keep 0.5
"""
import argparse, time, threading, statistics, math, json, sys

def measure():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        import pynvml
        pynvml.nvmlInit()
        HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        print("pynvml unavailable -> energy will be estimated from time only:", e)
        pynvml = None; HANDLE = None

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--keep", type=float, default=0.5,
                    help="fraction of MLP neurons kept (1.0=dense)")
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--prompt", default="Write a short paragraph about the ocean.")
    args = ap.parse_args()

    dtype = torch.float16
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {args.model} on {dev} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(dev).eval()

    # ---- install a top-k neuron mask on every MLP (LLaMA/Qwen-style SwiGLU) ----
    KEEP = {"v": 1.0}
    def patch_mlp(mlp):
        orig = mlp.forward
        def fwd(x):
            keep = KEEP["v"]
            if keep >= 1.0:
                return orig(x)
            gate = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)   # [.., I]
            I = gate.shape[-1]
            k = max(1, int(round(keep * I)))
            thr = torch.kthvalue(gate.abs(), I - k + 1, dim=-1, keepdim=True).values
            gate = gate * (gate.abs() >= thr)
            return mlp.down_proj(gate)
        mlp.forward = fwd
    n_patched = 0
    for m in model.modules():
        if all(hasattr(m, p) for p in ("gate_proj", "up_proj", "down_proj", "act_fn")):
            patch_mlp(m); n_patched += 1
    print(f"[patch] installed neuron-skip on {n_patched} MLP blocks")

    ids = tok(args.prompt, return_tensors="pt").to(dev)

    def power_sampler(stop, samples):
        while not stop.is_set():
            if HANDLE is not None:
                samples.append(pynvml.nvmlDeviceGetPowerUsage(HANDLE) / 1000.0)  # W
            time.sleep(0.01)

    def run_once(keep):
        KEEP["v"] = keep
        torch.cuda.synchronize() if dev == "cuda" else None
        stop = threading.Event(); samples = []
        th = threading.Thread(target=power_sampler, args=(stop, samples)); th.start()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.new_tokens,
                                  do_sample=False, use_cache=True)
        torch.cuda.synchronize() if dev == "cuda" else None
        dt = time.time() - t0
        stop.set(); th.join()
        n_new = out.shape[1] - ids.input_ids.shape[1]
        avg_w = statistics.mean(samples) if samples else float("nan")
        joules = avg_w * dt if samples else float("nan")
        return {"keep": keep, "seconds": dt, "new_tokens": int(n_new),
                "avg_watts": avg_w, "joules_per_token": joules / max(1, n_new),
                "tokens_per_sec": n_new / dt}

    # warmup
    for _ in range(2): run_once(1.0)

    def bench(keep):
        rows = [run_once(keep) for _ in range(args.repeats)]
        jpt = [r["joules_per_token"] for r in rows]
        wtt = [r["avg_watts"] for r in rows]
        tps = [r["tokens_per_sec"] for r in rows]
        return {"keep": keep,
                "joules_per_token_mean": statistics.mean(jpt),
                "joules_per_token_std": statistics.pstdev(jpt),
                "avg_watts_mean": statistics.mean(wtt),
                "tokens_per_sec_mean": statistics.mean(tps)}

    dense = bench(1.0)
    sparse = bench(args.keep)
    red = 100 * (1 - sparse["joules_per_token_mean"] / dense["joules_per_token_mean"])
    print(json.dumps({"dense": dense, "sparse": sparse,
                      "energy_per_token_reduction_pct": red}, indent=2))
    print(f"\n>>> keep={args.keep}: {red:.1f}% less energy PER TOKEN "
          f"({dense['joules_per_token_mean']:.3f} -> {sparse['joules_per_token_mean']:.3f} J/tok)")
    print(">>> Always pair this with the perplexity cost from measure_sparsity.py.")

if __name__ == "__main__":
    measure()

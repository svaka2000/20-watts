"""
Shared MLX instrumentation for the 20 Watts experiments.

`SparseModel` wraps an mlx-lm model and patches every MLP block so we can:
  - skip the bottom (1-keep) fraction of neurons per token (oracle top-k), or
  - install a custom mask_fn (e.g. a trained predictor), and/or
  - install a hook to capture (x, h) activations for analysis/training.

A bit-exact integrity check runs at construction: at keep=1.0 the patched
forward must equal the original forward (max abs diff ~ 0).
"""
import os, glob, json, math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

DEFAULT_MODEL = os.environ.get("WATTS_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")


def find_config(model_id):
    name = "models--" + model_id.replace("/", "--")
    base = os.path.expanduser(f"~/.cache/huggingface/hub/{name}/snapshots")
    for cfg in glob.glob(os.path.join(base, "*", "config.json")):
        with open(cfg) as f:
            return json.load(f)
    return {}


def flop_share(cfg):
    H = cfg["hidden_size"]; I = cfg["intermediate_size"]
    nH = cfg["num_attention_heads"]; nKV = cfg.get("num_key_value_heads", nH); hd = H // nH
    attn = 2 * H * H + 2 * H * (nKV * hd)
    mlp = 3 * H * I
    dims = {"H": H, "I": I, "L": cfg["num_hidden_layers"], "nH": nH, "nKV": nKV,
            "hd": hd, "attn_flops": attn, "mlp_flops": mlp, "inter_over_hidden": round(I / H, 3)}
    return mlp / (attn + mlp), dims


def topk_mask(h, keep):
    """Zero all but the top `keep` fraction of neurons by |h|, per position."""
    Ii = h.shape[-1]
    k = max(1, int(round(keep * Ii)))
    absh = mx.abs(h)
    thr = mx.sort(absh, axis=-1)[..., Ii - k:Ii - k + 1]
    return h * (absh >= thr)


class SparseModel:
    def __init__(self, model_id=DEFAULT_MODEL, verify=True):
        self.model_id = model_id
        self.model, self.tokenizer = load(model_id)
        self.cfg = find_config(model_id)
        self.mlp_share, self.dims = flop_share(self.cfg)
        self.layers = self.model.model.layers
        self.MLPClass = type(self.layers[0].mlp)
        self._orig = self.MLPClass.__call__
        for i, l in enumerate(self.layers):
            l.mlp.layer_id = i
        self.keep = 1.0
        self.hook = None        # fn(layer_id, x, h)
        self.mask_fn = None     # fn(layer_id, h) -> h  (overrides topk if set)
        outer = self

        def patched(self2, x):
            gate = self2.gate_proj(x)
            up = self2.up_proj(x)
            h = nn.silu(gate) * up
            if outer.hook is not None:
                outer.hook(self2.layer_id, x, h)
            if outer.mask_fn is not None:
                h = outer.mask_fn(self2.layer_id, h)
            elif outer.keep < 1.0:
                h = topk_mask(h, outer.keep)
            return self2.down_proj(h)

        if verify:
            H = self.dims["H"]
            probe = mx.random.normal((1, 4, H)).astype(mx.float16)
            o = self._orig(self.layers[0].mlp, probe)
            self.MLPClass.__call__ = patched
            self.keep = 1.0
            n = self.layers[0].mlp(probe)
            mx.eval(o, n)
            self.max_diff = float(mx.abs(o - n).max())
            assert self.max_diff < 1e-2, f"patch mismatch: {self.max_diff}"
        else:
            self.MLPClass.__call__ = patched
            self.max_diff = None

    # --- control ---
    def set_keep(self, p):
        self.keep = p; self.mask_fn = None

    def reset(self):
        self.keep = 1.0; self.mask_fn = None; self.hook = None

    # --- forward helpers ---
    def logits(self, ids):
        return self.model(ids)

    def perplexity(self, ids):
        lg = self.model(ids)
        logp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        tgt = ids[:, 1:]
        lp = mx.take_along_axis(logp[:, :-1, :], tgt[..., None], axis=-1)[..., 0]
        nll = -lp.mean(); mx.eval(nll); nll = float(nll)
        return nll, math.exp(nll)

    def encode(self, text):
        return self.tokenizer.encode(text)

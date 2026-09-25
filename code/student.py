"""Student model: a pre-norm decoder with RMSNorm, RoPE and SwiGLU.

Everything here is switched by config keys, so an ablation is just another
config file rather than a code fork:

    vocab, context, width, heads, depth   -- as in configs/baseline.json
    norm  : 'rms'   | 'layer'      (default 'rms')
    mlp   : 'swiglu'| 'gelu'       (default 'swiglu')
    pos   : 'rope'  | 'learned'    (default 'rope')
    bias  : bool                   (default False)
    dropout : float                (default 0.0)
    mlp_hidden : int               (default round(8/3 * width), SwiGLU only)

Relationship to `model.py`: the data flow, pre-norm placement, tied output
embedding and causal attention are unchanged. The four edits are the
normalization type, the gated MLP, the position encoding, and the removal of
linear biases. Keeping the rest identical is what makes the ablation
interpretable.

AI assistance is disclosed in the repository README; see also `SETUP_NOTES.md`.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def rotate_half(x):
    """Split the last dimension in half and rotate: [a, b] -> [-b, a]."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_cache(head_dim, length, device, dtype):
    """cos/sin tables of shape [length, head_dim] for rotary position embedding."""
    inverse = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
                               / head_dim))
    angles = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inverse)
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


class RMSNorm(nn.Module):
    """LayerNorm without mean subtraction, using only a per-channel scale."""

    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    """Gated MLP: down(silu(gate(x)) * up(x)). Three matrices instead of two."""

    def __init__(self, width, hidden, bias):
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=bias)
        self.up = nn.Linear(width, hidden, bias=bias)
        self.down = nn.Linear(hidden, width, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def build_mlp(kind, width, hidden, bias):
    if kind == 'swiglu':
        return SwiGLU(width, hidden, bias)
    if kind == 'gelu':
        # Same width as the baseline MLP: 4 * width hidden units, GELU activation.
        return nn.Sequential(nn.Linear(width, 4 * width, bias=bias), nn.GELU(),
                             nn.Linear(4 * width, width, bias=bias))
    raise ValueError(f'unknown mlp kind: {kind}')


def build_norm(kind, width):
    if kind == 'rms':
        return RMSNorm(width)
    if kind == 'layer':
        return nn.LayerNorm(width)
    raise ValueError(f'unknown norm kind: {kind}')


class Block(nn.Module):
    def __init__(self, config, width):
        super().__init__()
        self.heads = config['heads']
        self.head_dim = width // self.heads
        self.pos = config['pos']
        self.dropout = config['dropout']
        self.norm1 = build_norm(config['norm'], width)
        self.norm2 = build_norm(config['norm'], width)
        self.qkv = nn.Linear(width, 3 * width, bias=config['bias'])
        self.proj = nn.Linear(width, width, bias=config['bias'])
        self.mlp = build_mlp(config['mlp'], width, config['mlp_hidden'], config['bias'])
        self.drop = nn.Dropout(config['dropout'])
        self._cache = {}

    def rope(self, length, device, dtype):
        key = (length, device, dtype)
        tables = self._cache.get(key)
        if tables is None:
            tables = rope_cache(self.head_dim, length, device, dtype)
            self._cache[key] = tables
        return tables

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = (self.qkv(self.norm1(x))
                   .view(batch, length, 3, self.heads, self.head_dim)
                   .permute(2, 0, 3, 1, 4))
        if self.pos == 'rope':
            cos, sin = self.rope(length, x.device, q.dtype)
            cos, sin = cos.view(1, 1, length, self.head_dim), sin.view(1, 1, length, self.head_dim)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
        # Each position attends only to itself and earlier input tokens.
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        return x + self.drop(self.mlp(self.norm2(x)))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Fill in the optional switches so a minimal config (e.g. the contract
        # tests) still builds the modern architecture.
        config = dict(config)
        width = config['width']
        config.setdefault('norm', 'rms')
        config.setdefault('mlp', 'swiglu')
        config.setdefault('pos', 'rope')
        config.setdefault('bias', False)
        config.setdefault('dropout', 0.0)
        config.setdefault('mlp_hidden', int(round(8 / 3 * width)))
        if not 0.0 <= config['dropout'] < 1.0:
            raise ValueError('dropout must be in [0, 1).')
        self.config = config
        self.context = config['context']
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = nn.Embedding(self.context, width) if config['pos'] == 'learned' else None
        self.drop = nn.Dropout(config['dropout'])
        self.blocks = nn.ModuleList([Block(config, width) for _ in range(config['depth'])])
        self.norm = build_norm(config['norm'], width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.drop(self.token(ids))
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalized next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Evaluation interface: normalized log probabilities, no access to targets.

        Every call is independent: nothing is carried between windows, so a
        prediction at position t depends only on ids[:, :t+1].
        """
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return GPT(config)

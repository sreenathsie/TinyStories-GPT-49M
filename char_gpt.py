"""
Character-level GPT (nanoGPT-style) - runs end-to-end on free Colab T4 in ~10-15 min.

IMPORTANT: Before running, set Runtime -> Change runtime type -> T4 GPU -> Save.
This script will refuse to train on CPU by default (it's way too slow) unless
you explicitly allow it below.

Paste this whole file into one Colab cell, or upload+run: !python char_gpt.py
"""

import os
import time
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 0. Setup + GPU check
# ---------------------------------------------------------------------------
torch.manual_seed(1337)

ALLOW_CPU = False  # set True only if you really want to train on CPU (very slow)

if torch.cuda.is_available():
    device = "cuda"
    print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
else:
    device = "cpu"
    print("WARNING: No GPU detected, using CPU.")
    print("Fix: Colab menu -> Runtime -> Change runtime type -> T4 GPU -> Save, then rerun.")
    if not ALLOW_CPU:
        raise SystemExit(
            "Stopping because no GPU is available and ALLOW_CPU=False. "
            "Switch to a GPU runtime and rerun this cell."
        )

# ---------------------------------------------------------------------------
# 1. Dataset - tiny Shakespeare (1MB, classic for this exercise)
# ---------------------------------------------------------------------------
data_path = "input.txt"
if not os.path.exists(data_path):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, data_path)

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Dataset length: {len(text):,} characters")

# ---------------------------------------------------------------------------
# 2. "Tokenizer" - character level. Every unique char = one token.
#    Simplest possible tokenizer: no BPE, no merges, just a lookup table.
#    Real LLMs use BPE - this is for learning the core architecture first.
# ---------------------------------------------------------------------------
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size} unique characters")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ---------------------------------------------------------------------------
# 3. Hyperparameters
# ---------------------------------------------------------------------------
block_size = 256       # context length
batch_size = 64
d_model = 256
n_head = 8
n_layer = 6
dropout = 0.1
max_iters = 8000       # increased from 3000 - will resume from your saved checkpoint
eval_interval = 300
eval_iters = 50        # reduced from 100 for faster feedback
learning_rate = 3e-4
ckpt_every = 500

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---------------------------------------------------------------------------
# 4. Model - decoder-only transformer with causal self-attention
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class MLP(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CharGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model params: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ---------------------------------------------------------------------------
# 5. Loss estimation helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ---------------------------------------------------------------------------
# 6. Build model, optimizer, resume from checkpoint if present
# ---------------------------------------------------------------------------
LOCAL_CKPT = "char_gpt_ckpt.pt"

model = CharGPT(vocab_size, d_model, n_head, n_layer, block_size, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

start_iter = 0
if os.path.exists(LOCAL_CKPT):
    print("Resuming from checkpoint...")
    ckpt = torch.load(LOCAL_CKPT, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_iter = ckpt["iter"] + 1

# ---------------------------------------------------------------------------
# 7. Training loop
# ---------------------------------------------------------------------------
t0 = time.time()
for iter in range(start_iter, max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        elapsed = time.time() - t0
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
              f"elapsed {elapsed/60:.1f} min")

    if iter % ckpt_every == 0 and iter > 0:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iter": iter}, LOCAL_CKPT)
        print(f"  checkpoint saved at step {iter}")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iter": max_iters - 1}, LOCAL_CKPT)
print(f"\nTraining done in {(time.time()-t0)/60:.1f} minutes")

# ---------------------------------------------------------------------------
# 8. Generate sample text
# ---------------------------------------------------------------------------
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)
print("\n--- Generated text ---\n")
print(decode(generated[0].tolist()))

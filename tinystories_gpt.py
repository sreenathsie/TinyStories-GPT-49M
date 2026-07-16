"""
Word-piece (BPE) GPT trained on TinyStories.
Uses the REAL TinyStories dataset via the `datasets` library (streaming + retries) -
no synthetic/dummy fallback, so you always know you're training on real data.
Includes: early stopping, live progress bar with ETA, multi-prompt generation.
"""

import os
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"

import time
import subprocess
import sys

# ---------------------------------------------------------------------------
# 0. Install dependencies
# ---------------------------------------------------------------------------
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets", "tiktoken", "tqdm"], check=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from tqdm.auto import tqdm
from datasets import load_dataset

# ---------------------------------------------------------------------------
# 1. Setup + GPU check
# ---------------------------------------------------------------------------
torch.manual_seed(1337)

if torch.cuda.is_available():
    device = "cuda"
    print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
else:
    raise SystemExit(
        "No GPU detected. Go to Runtime -> Change runtime type -> T4 GPU -> Save, then rerun."
    )

# ---------------------------------------------------------------------------
# 2. Tokenizer
# ---------------------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
EOT = enc.eot_token
print(f"Tokenizer vocab size: {vocab_size}")

# ---------------------------------------------------------------------------
# 3. Dataset - REAL TinyStories only. If this fails, it fails loudly (raises),
#    rather than silently swapping in fake template data. If you hit repeated
#    failures, it's a genuine network/HF issue - wait and retry, don't proceed
#    on fake data.
# ---------------------------------------------------------------------------
NUM_STORIES = 60_000
CACHE_PATH = "tinystories_tokens.pt"   # point this at your Drive path if you want persistence

if os.path.exists(CACHE_PATH):
    print("Loading cached tokenized data...")
    all_ids = torch.load(CACHE_PATH)
else:
    print(f"Streaming {NUM_STORIES:,} stories from TinyStories...")

    max_retries = 5
    all_ids = []
    for attempt in range(1, max_retries + 1):
        try:
            ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            all_ids = []
            t0 = time.time()
            for i, example in enumerate(ds):
                if i >= NUM_STORIES:
                    break
                ids = enc.encode_ordinary(example["text"])
                all_ids.extend(ids)
                all_ids.append(EOT)
                if i % 10_000 == 0:
                    print(f"  tokenized {i:,} stories, {len(all_ids):,} tokens so far "
                          f"({time.time()-t0:.0f}s elapsed)")
            break  # success
        except Exception as e:
            print(f"  Attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                raise RuntimeError(
                    "Could not download real TinyStories data after several retries. "
                    "This is a Hugging Face / network issue - wait a few minutes and rerun. "
                    "Not falling back to fake data."
                )
            wait = 5 * attempt
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)

    all_ids = torch.tensor(all_ids, dtype=torch.long)
    torch.save(all_ids, CACHE_PATH)

print(f"\nTotal tokens: {len(all_ids):,}")

n = int(0.98 * len(all_ids))
train_data = all_ids[:n]
val_data = all_ids[n:]

# ---------------------------------------------------------------------------
# 4. Hyperparameters
# ---------------------------------------------------------------------------
block_size = 256
batch_size = 16
grad_accum_steps = 4
d_model = 384
n_head = 6
n_layer = 6
dropout = 0.1
max_iters = 7000
eval_interval = 350
eval_iters = 50
learning_rate = 3e-4
ckpt_every = 500

# Early stopping
early_stop_patience = 5   # stop if val loss doesn't improve for 5 consecutive checks
patience_counter = 0

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---------------------------------------------------------------------------
# 5. Model
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


class GPT(nn.Module):
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
# 6. Loss estimation
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
# 7. Build model, optimizer, resume from checkpoint if present
# ---------------------------------------------------------------------------
LOCAL_CKPT = "tinystories_gpt_ckpt.pt"
BEST_CKPT = "tinystories_gpt_best.pt"

model = GPT(vocab_size, d_model, n_head, n_layer, block_size, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

start_iter = 0
best_val_loss = float("inf")
if os.path.exists(LOCAL_CKPT):
    print("Resuming from checkpoint...")
    ckpt = torch.load(LOCAL_CKPT, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_iter = ckpt["iter"] + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))

# ---------------------------------------------------------------------------
# 8. Training loop - tqdm progress bar + early stopping
# ---------------------------------------------------------------------------
t0 = time.time()
current_train_loss = None
early_stop_triggered = False

pbar = tqdm(range(start_iter, max_iters), initial=start_iter, total=max_iters,
            desc="Training", unit="step")
for iter in pbar:
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        current_train_loss = losses["train"]
        elapsed = time.time() - t0
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
              f"elapsed {elapsed/60:.1f} min")

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save({"model": model.state_dict(), "iter": iter, "val_loss": best_val_loss}, BEST_CKPT)
            print(f"  new best val loss, saved to {BEST_CKPT}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience and iter > 1000:
            print(f"\nEarly stopping: val loss hasn't improved for {early_stop_patience} checks. "
                  f"Best val loss: {best_val_loss:.4f}")
            early_stop_triggered = True
            break

    if iter % ckpt_every == 0 and iter > 0:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": iter,
            "best_val_loss": best_val_loss,
        }, LOCAL_CKPT)

    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(grad_accum_steps):
        xb, yb = get_batch("train")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(xb, yb)
            loss = loss / grad_accum_steps
        loss.backward()
    optimizer.step()

    pbar.set_postfix({
        "train_loss": f"{current_train_loss:.3f}" if current_train_loss is not None else "...",
        "best_val": f"{best_val_loss:.3f}" if best_val_loss < float("inf") else "...",
    })

if not early_stop_triggered:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": max_iters - 1,
        "best_val_loss": best_val_loss,
    }, LOCAL_CKPT)

print(f"\nTraining done in {(time.time()-t0)/60:.1f} minutes. Best val loss: {best_val_loss:.4f}")

# ---------------------------------------------------------------------------
# 9. Generate sample text from multiple prompts, using the BEST checkpoint
# ---------------------------------------------------------------------------
print("\nLoading best checkpoint for generation...")
best = torch.load(BEST_CKPT, map_location=device)
model.load_state_dict(best["model"])
model.eval()

prompts = ["Once upon a time", "The little girl", "One sunny day", "In the forest"]

print("\n" + "=" * 60)
print("Generated Stories")
print("=" * 60)
for prompt in prompts:
    print(f"\nPrompt: '{prompt}'")
    print("-" * 40)
    context = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=150, temperature=0.8, top_k=40)
    print(enc.decode(generated[0].tolist()))
    print("-" * 40)

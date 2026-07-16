# TinyStories-GPT-49M

A 49-million-parameter GPT-style language model, built from scratch and trained on free Google Colab (single T4 GPU). Built as a hands-on project to learn transformer architecture, tokenization, and the LLM training pipeline end-to-end.

## What this model is

A **decoder-only transformer language model** (GPT architecture), trained from random initialization — no pretrained weights used. It's a *base* language model: it predicts the next token in a sequence of text. It is **not instruction-tuned**, so it continues text in the style of its training data rather than answering questions directly.

## What it does

Given a text prompt, the model generates a continuation in the style of a children's short story — consistent character names, simple plots, age-appropriate vocabulary and grammar. For example, given `"Once upon a time"`, it generates a full short story with a beginning, middle, and end.

It does **not** function as a chatbot or assistant. Asking it a direct question (e.g. "who is Lily?") will not produce a direct answer — it will produce more story text, because that is the only task it was trained on.

## Real-world use cases

- **Educational tool** for understanding how LLMs work internally (attention, tokenization, training loops) at a scale that's cheap and fast to experiment with
- **Base model for further fine-tuning** — e.g. instruction-tuning on Q&A pairs to turn it into a simple chatbot (a natural next step, not yet done)
- **Children's story generation** as a creative writing aid or prompt-starter
- **Baseline/reference point** for comparing architecture or training changes at small scale before committing to larger, more expensive training runs

It is explicitly **not** suitable for factual question-answering, reasoning tasks, or any production use case — it is a learning-scale model with a narrow, single-purpose training objective.

## Architecture

| Component | Detail |
|---|---|
| Type | Decoder-only transformer (GPT-style) |
| Parameters | 49.33M |
| Layers | 6 |
| Attention heads | 6 |
| Embedding dimension (d_model) | 384 |
| Context length (block_size) | 256 tokens |
| Normalization | Pre-norm LayerNorm |
| Feedforward | 4x expansion, GELU activation |
| Attention implementation | PyTorch `scaled_dot_product_attention` (fused, flash-attention-like) |
| Positional encoding | Learned absolute position embeddings |

## Tokenizer

- **GPT-2's BPE tokenizer**, reused via `tiktoken` (`gpt2` encoding), vocab size 50,257
- Not trained from scratch — reusing a proven tokenizer avoided an extra pipeline stage and is standard practice for a first from-scratch model
- An earlier version of this project used a **character-level tokenizer** (one token per character) on a Shakespeare dataset; it was replaced with BPE specifically because the character-level model would invent nonsense words (e.g. "straitiff", "unlaugh") since it had no concept of whole words. BPE fixed this.

## Dataset

- **[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)** (Eldan & Li, Microsoft Research) — a synthetic dataset of simple short stories using a restricted vocabulary, designed specifically for training small language models to produce coherent text.
- Used a streamed subset of **60,000 stories** (~13.3M tokens after tokenization), not the full dataset.
- 98% train / 2% validation split.

### Why TinyStories, and why not the full dataset

An earlier version of this project trained on the full works of Shakespeare (~1.1M characters). At that scale, the model **overfit**: training loss kept dropping while validation loss got worse, because the model started memorizing the training text instead of generalizing. TinyStories' much larger size and simpler, repetitive-but-varied structure let the model keep improving on unseen validation data throughout the full training run — no overfitting was observed even after 7,000 training steps.

## How it was trained

- **Hardware**: single NVIDIA T4 GPU (Google Colab free tier)
- **Precision**: mixed precision (bfloat16 autocast) to reduce memory usage
- **Batch size**: 16 per micro-batch, with 4 steps of gradient accumulation → effective batch size 64
  - (Straight batch_size=64 caused CUDA out-of-memory errors, because the 50k-token vocabulary makes the output logits tensor large; gradient accumulation gives the same effective batch size at a fraction of the peak memory.)
- **Optimizer**: AdamW, learning rate 3e-4, weight decay 0.1
- **Checkpointing**: saved every 500 steps, plus a separate "best" checkpoint saved whenever validation loss improved (so the best-generalizing model is always recoverable, even if later training overfits)
- **Early stopping**: configured to stop if validation loss failed to improve for 5 consecutive evaluation checks (did not trigger in the final run — validation loss was still improving at the final step)
- **Total training**: 7,000 steps, done in two resumed sessions across multiple Colab runtimes (using checkpoint resume, since free Colab sessions are time-limited)

### Training results

| Step | Train loss | Val loss |
|---|---|---|
| 0 | 10.90 | 10.91 |
| 1,750 | 2.23 | 2.27 |
| 3,500 | 1.89 | 1.99 |
| 5,250 | 1.75 | 1.88 |
| 6,650 | 1.65 | **1.85** (best) |
| 6,999 (final) | 1.66 | 1.87 |

- **Best validation loss: 1.8467** (step 6,650)
- **Total training time: 180.4 minutes** (~3 hours) for the final 7,000-step run
- Validation loss was still improving at the end of training — the model was not trained to convergence, and more steps and/or more data would likely improve it further

## Methods used

- Decoder-only transformer built from scratch in PyTorch (no pretrained model weights)
- BPE tokenization via `tiktoken`, reusing GPT-2's tokenizer
- Streaming dataset loading (`datasets` library) to avoid downloading the full dataset into memory
- Mixed-precision training (`torch.autocast`, bfloat16)
- Gradient accumulation to work around single-GPU memory limits
- Fused scaled-dot-product attention for speed
- Checkpoint-and-resume training to work within free Colab's session time limits
- Best-checkpoint tracking based on validation loss, separate from the most-recent checkpoint, to guard against overfitting
- Early stopping (configured, not triggered in this run)

## Limitations / disadvantages

- **Not a chatbot.** It is a base language model trained only on next-token prediction over story text. It does not understand or answer questions — see the chat transcript below for a concrete example of this limitation.
- **Small scale.** 49M parameters and ~13M training tokens is far below the data/parameter scale needed for general-purpose language understanding (for reference, Chinchilla scaling guidelines suggest a 49M model would ideally see ~1B+ tokens, not 13M).
- **Narrow domain.** Trained exclusively on simple children's stories; it has no knowledge of facts, reasoning, code, or any topic outside that narrow style and vocabulary.
- **Occasional logical errors** in generated text (e.g. incorrect pronoun/subject attribution within a sentence), typical of small-scale models.
- **No safety or content filtering** beyond what's inherent in the TinyStories dataset itself; not evaluated for bias or harmful outputs.
- **Not trained to convergence** — validation loss was still decreasing when training stopped, so the checkpoint provided is a snapshot of progress, not a final, fully optimized result.

### Example: base model vs. chatbot expectation

```
You: who is lily
GPT: started to cry and sneezed so hard that it made her bleed.
Lily's mom took her to the doctor and doctor said it was okay...
```
The model responds with more story text rather than an answer, because it was never trained on question-answer pairs — only on continuing story-style text.

## Advantages

- Fully transparent, from-scratch implementation — every component (attention, embeddings, training loop) is visible and understandable, unlike using a pretrained model as a black box
- Trains end-to-end on **free** compute (Colab free tier, single T4)
- Fast to iterate on: full training run completes in hours, not days, enabling quick experimentation
- Demonstrates a complete, correct LLM training pipeline: tokenizer choice, dataset handling, architecture, mixed precision, gradient accumulation, checkpointing, early stopping, and generation
- Produces genuinely coherent, grammatically correct short-story text despite its small size and limited training data
- Serves as a solid, well-documented base for further work (see below)

## Future goals

- **Instruction-tuning**: fine-tune this base model on question-answer pairs so it can function as a simple chatbot, rather than only continuing text
- **Scale up**: move toward a larger target (originally ~300M parameters) using the same pipeline, with more training data and longer training
- **Train tokenizer from scratch** on the target dataset rather than reusing GPT-2's tokenizer, for a vocabulary better matched to the training domain
- **Broader/larger dataset** to reduce the current narrow-domain limitation and improve generalization
- **Longer training with more data** to reach convergence rather than stopping while validation loss is still improving
- **Evaluation beyond loss**: add qualitative/quantitative benchmarks beyond validation loss (e.g. human evaluation of coherence, simple QA accuracy after instruction-tuning)

## Repository structure

```
.
├── README.md
├── char_gpt.py              # Stage 1: character-level GPT on Shakespeare (learning exercise)
├── tinystories_gpt.py        # Stage 2: BPE tokenizer + TinyStories (final model)
└── notebook/
    └── training_notebook.ipynb   # Full Colab session, both stages
```

## How to reproduce

1. Open in Google Colab, set **Runtime → Change runtime type → T4 GPU**
2. Run `tinystories_gpt.py` (or the notebook) — dependencies install automatically
3. Training resumes automatically from a checkpoint if one exists in the working directory
4. Trained weights can be saved to Google Drive for persistence across Colab sessions (see save-checkpoint cell)

## Acknowledgements

- [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) — Eldan & Li, Microsoft Research
- Architecture and training loop style inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) (Andrej Karpathy)
- GPT-2 tokenizer via [tiktoken](https://github.com/openai/tiktoken) (OpenAI)
## Screenshot of output
![image alt](https://github.com/sreenathsie/TinyStories-GPT-49M/blob/452606ec94867cb353a2113451131e3d47283f92/Screenshot%202026-07-16%20185401.png)

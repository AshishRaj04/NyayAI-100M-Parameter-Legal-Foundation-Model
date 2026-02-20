# Project NyayAI

**A 103M Parameter LLM Trained from Scratch on Indian Legal Data**

NyayAI is a custom-built GPT-style language model trained entirely from scratch on 269 million tokens of Indian Supreme Court and High Court judgments. No pre-trained weights, no fine-tuning — every parameter was learned from raw legal text.

---

## 🎯 The Problem

India's legal system faces a staggering backlog of over **5 crore pending cases**. This judicial pendency causes inordinate delays, denying timely justice to millions. Legal research, case analysis, and document drafting remain extremely time-intensive bottlenecks.

## 💡 The Solution

NyayAI is a foundational step toward AI-assisted legal intelligence. It's a **specialist model** — built from the ground up to understand and generate text in the language and structure of Indian law.

This repository contains the **complete, end-to-end pipeline**: raw data processing → tokenization → model architecture → distributed GPU training → local inference → web UI.

---

## ✨ Key Highlights

- **Built from scratch** — Custom GPT architecture implemented in PyTorch, no HuggingFace dependencies
- **103M parameters** — 9-layer transformer with 12 attention heads and 768-dim embeddings
- **269M training tokens** — 1.25 GB of cleaned Indian legal judgments (Supreme Court + High Courts)
- **Weight tying** — Token embedding weights shared with output head, reducing parameter count
- **Cosine LR schedule** — Warmup + cosine decay for stable training
- **Fault-tolerant training** — Per-epoch checkpointing with auto-download and resumable training
- **Runs locally** — Fast CPU inference (~10 tokens/sec), no GPU needed for generation
- **Web interface** — Dark-themed premium UI with generation controls

---

## 🧠 Model Architecture

```
Input Tokens
    ↓
Token Embedding (50,257 × 768) + Positional Embedding (512 × 768)
    ↓
┌─────────────────────────────────────┐
│  Transformer Block (×9)              │
│  ├── LayerNorm                       │
│  ├── Multi-Head Attention (12 heads) │
│  │   ├── Q, K, V projections (768)   │
│  │   ├── Causal mask                 │
│  │   └── Output projection          │
│  ├── Residual connection + Dropout   │
│  ├── LayerNorm                       │
│  ├── Feed-Forward (768 → 3072 → 768) │
│  └── Residual connection + Dropout   │
└─────────────────────────────────────┘
    ↓
Final LayerNorm
    ↓
Output Head (weight-tied with Token Embedding)
    ↓
Logits (50,257)
```

| Parameter           | Value                           |
| ------------------- | ------------------------------- |
| Vocabulary Size     | 50,257 (GPT-2 BPE via tiktoken) |
| Context Length      | 512 tokens                      |
| Embedding Dimension | 768                             |
| Attention Heads     | 12 (head dim = 64)              |
| Transformer Layers  | 9                               |
| Feed-Forward Hidden | 3,072 (4× emb_dim)              |
| Dropout Rate        | 0.1                             |
| Total Parameters    | **102,762,240 (~103M)**         |
| Model Size (FP32)   | **392 MB**                      |
| Weight Tying        | Yes (tok_emb ↔ out_head)        |

---

## 📊 Training Details

### Infrastructure

| Component         | Details                                            |
| ----------------- | -------------------------------------------------- |
| GPU               | NVIDIA A100 (40 GB) via [Modal](https://modal.com) |
| Framework         | PyTorch 2.x                                        |
| Tokenizer         | tiktoken (GPT-2 BPE, 50,257 tokens)                |
| Training Platform | Modal (serverless GPU cloud)                       |

### Training Configuration

| Setting                 | Value                                     |
| ----------------------- | ----------------------------------------- |
| Batch Size              | 8,192 tokens (context_length × batch)     |
| Total Training Tokens   | 269,098,817 (~269M)                       |
| Train/Val Split         | 90% / 10%                                 |
| Train Batches per Epoch | 29,564                                    |
| Optimizer               | AdamW (β1=0.9, β2=0.99, ε=1e-8, wd=0.1)   |
| Peak Learning Rate      | 4e-4                                      |
| Min Learning Rate       | 4e-5                                      |
| LR Schedule             | Linear warmup (2000 steps) → Cosine decay |
| Gradient Clipping       | 1.0 (global norm)                         |
| Epochs                  | 5 (in progress)                           |

### Training Progress (Epoch 1 / 5)

| Metric           | Value                    |
| ---------------- | ------------------------ |
| Training Time    | 223 minutes (~3.7 hours) |
| Starting Loss    | 495.7                    |
| Final Train Loss | 2.760                    |
| Final Val Loss   | 2.674                    |
| Tokens Processed | 242M                     |
| Checkpoint Size  | 1,185 MB                 |

> Loss dropped from **495 → 2.67** in a single epoch across 29,564 batches. Remaining 4 epochs will further improve quality.

### Fault-Tolerant Training Features

- **Per-epoch checkpointing** — Model + optimizer state saved after every epoch
- **Auto-download** — Checkpoints automatically downloaded from cloud to local machine
- **Resumable training** — Pass `--resume-from` flag to continue from any checkpoint
- **Per-epoch logs** — Training metrics saved as JSON after each epoch (no data loss on interruption)
- **Generator pattern** — `remote_gen()` streams results to client as epochs complete

---

## 📂 Project Structure

```
LLM-FROM-SCRATCH/
│
├── llm_engine.py              # GPT model architecture (Transformer, MHA, FFN)
├── data_loader.py             # Dataset/DataLoader (chunked tokenization)
├── data_cleaner.py            # Raw legal text cleaning pipeline
├── training.py                # Modal-based distributed training script
├── count_params.py            # Parameter counting utility
│
├── inference/                 # Inference & web UI
│   ├── infer.py               # Local inference engine (standalone or importable)
│   ├── app.py                 # Flask web server
│   └── templates/
│       └── index.html         # Dark-themed web interface
│
├── checkpoints/               # Model checkpoints (not in git)
│   ├── epoch_1_model_and_optimizer.pth
│   └── training_log_epoch_1.json
│
├── data/                      # Training corpus
│   └── combined_legal_data.txt
│
├── logs/                      # Raw training logs
│   └── epoch_1_logs.txt
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/your-username/LLM-FROM-SCRATCH.git
cd LLM-FROM-SCRATCH

python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 2. Run Inference (CLI)

```bash
python inference/infer.py --prompt "The verdict of the court is " --max-tokens 200
```

### 3. Run Web Server

```bash
python inference/app.py
# Open http://localhost:5000
```

### 4. Train the Model (requires Modal account)

```bash
# First time - full training
modal run training.py --num-epochs 5 --eval-freq 50 --eval-iter 5

# Resume from checkpoint
modal run training.py --num-epochs 5 --resume-from runs/20260219-184305/epoch_1_model_and_optimizer.pth
```

---

## 🛠️ Tech Stack

| Layer                   | Technology                  |
| ----------------------- | --------------------------- |
| Model Architecture      | Custom GPT (PyTorch)        |
| Tokenizer               | tiktoken (GPT-2 BPE)        |
| Training Infrastructure | Modal (serverless A100 GPU) |
| Inference               | PyTorch (CPU)               |
| Web Backend             | Flask                       |
| Web Frontend            | Vanilla HTML/CSS/JS         |
| Data                    | 1.25 GB Indian legal corpus |

---

## 📈 Sample Output

**Prompt:** `Under Section 498A of the Indian Penal Code,`

**Generated (Epoch 1):**

> Under Section 498A of the Indian Penal Code, as per the document, the charge under Section 376 IPC was based on the complaint, which was filed by the appellant before the trial court.

> _Note: After epoch 1, the model generates coherent legal English but may reference incorrect sections. Quality improves significantly with continued training (epochs 2-5)._

---

## 🔮 Roadmap

- [x] Custom GPT architecture from scratch
- [x] 103M parameter model training
- [x] Per-epoch checkpointing & resumable training
- [x] Local CPU inference engine
- [x] Web interface with generation controls
- [ ] Complete 5-epoch training
- [ ] RAG integration for grounded legal answers
- [ ] Fine-tuning for instruction-following
- [ ] Deployment to production

---

## 👨‍💻 Author

**Ashish Raj**

Built as a proof-of-concept for AI-powered legal intelligence in India.

---

## 📄 License

This project is for educational and research purposes. The training data consists of publicly available Indian court judgments.

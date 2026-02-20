from llm_engine import GPTModel

MODEL_CONFIG = {
    "vocab_size": 50257,
    "context_length": 512,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 9,
    "drop_rate": 0.1,
    "kqv_bias": False,
}

model = GPTModel(MODEL_CONFIG)

print("=" * 60)
print(f"{'Layer':<40} {'Params':>10}")
print("=" * 60)

total = 0
for name, param in model.named_parameters():
    count = param.numel()
    total += count
    print(f"{name:<40} {count:>10,}")

print("=" * 60)
print(f"{'TOTAL':<40} {total:>10,}")
print(f"{'Model size (FP32)':<40} {total * 4 / (1024**2):>8.2f} MB")
print("=" * 60)

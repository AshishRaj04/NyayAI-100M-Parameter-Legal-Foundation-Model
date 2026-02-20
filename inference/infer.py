"""
Inference module for NyayAI - loads trained GPT model and generates text.
Can be run standalone or imported by app.py for the Flask server.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import tiktoken
from llm_engine import GPTModel

DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "epoch_1_model_and_optimizer.pth"
)

MODEL_CONFIG = {
    "vocab_size": 50257,
    "context_length": 512,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 9,
    "drop_rate": 0.1,
    "kqv_bias": False,
}

EOT_TOKEN = "<|" + "endoftext" + "|>"


class NyayAIInference:
    """Handles loading the model and generating text."""

    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tiktoken.get_encoding("gpt2")

        print(f"Loading model from {checkpoint_path}...")
        self.model = GPTModel(MODEL_CONFIG)

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        state_dict = checkpoint["model_state_dict"]
        # Handle weight tying: out_head shares weights with tok_emb
        if "out_head.weight" in state_dict and "tok_emb.weight" in state_dict:
            if torch.equal(state_dict["out_head.weight"], state_dict["tok_emb.weight"]):
                del state_dict["out_head.weight"]

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded: {total_params / 1e6:.1f}M params on {self.device}")

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=256, temperature=0.8, top_k=40):
        """Generate text continuation from a prompt."""
        encoded = self.tokenizer.encode(prompt, allowed_special={EOT_TOKEN})
        idx = torch.tensor(encoded, device=self.device).unsqueeze(0)
        context_size = self.model.pos_emb.weight.shape[0]

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits = self.model(idx_cond)
            logits = logits[:, -1, :]

            # Top-k filtering
            if top_k is not None and top_k > 0:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float("-inf"), device=logits.device),
                    logits,
                )

            # Temperature scaling + sampling
            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            idx = torch.cat((idx, idx_next), dim=1)

            # Stop at EOT token
            eot_id = self.tokenizer.encode(EOT_TOKEN, allowed_special={EOT_TOKEN})[0]
            if idx_next.item() == eot_id:
                break

        generated = idx.squeeze(0).tolist()
        text = self.tokenizer.decode(generated)
        # Strip the EOT token from display
        text = text.replace(EOT_TOKEN, "").strip()
        return text


# ---- Standalone CLI ----
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NyayAI Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The verdict of the court is ",
        help="Prompt to generate from",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    engine = NyayAIInference(checkpoint_path=args.checkpoint)

    print("\n" + "=" * 60)
    print(f"Prompt: {args.prompt}")
    print("=" * 60)

    output = engine.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(output)
    print("=" * 60)

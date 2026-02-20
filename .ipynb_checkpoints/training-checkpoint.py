import os
import time
import modal

app = modal.App("LLM_Training")

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("llm_engine")
    .add_local_python_source("data_loader")
    .add_local_file("./data/cleaned_legal_data.csv", remote_path="/app/cleaned_legal_data.csv")
)

CHECKPOINTS_VOL = modal.Volume.from_name("llm_checkpoints", create_if_missing=True)


@app.cls(
    image=image,
    gpu="A10",
    secrets=[modal.Secret.from_name("llm_training_secret")],
    volumes={"/checkpoints": CHECKPOINTS_VOL},
    timeout=9000 # 150 minutes
)
class TrainingScript:
    @modal.enter()
    def setup(self):
        import torch
        import tiktoken
        import pandas as pd

        from data_loader import create_dataloader_v1
        from llm_engine import createGPTModel

        self.torch = torch
        self.tiktoken = tiktoken
        self.pd = pd
        self.create_dataloader_v1 = create_dataloader_v1
        self.createGPTModel = createGPTModel

        # Model config
        self.MODEL_CONFIG = {
            "vocab_size": 50257,
            "context_length": 256,
            "emb_dim": 384,
            "n_heads": 12,
            "n_layers": 6,
            "drop_rate": 0.1,
            "kqv_bias": False,
        }

        # Tokenizer
        self.tokenizer = self.tiktoken.get_encoding("gpt2")

        # Load data
        file_path = "/app/cleaned_legal_data.csv"
        df = self.pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} cases")
        text_data = " ".join(df["cleaned_text"].astype(str))

        total_characters = len(text_data)
        total_tokens = len(self.tokenizer.encode(text_data))
        print(f"Characters: {total_characters}")
        print(f"Tokens: {total_tokens}")

        # Train/val split
        train_ratio = 0.9
        split_idx = int(train_ratio * len(text_data))
        train_data = text_data[:split_idx]
        val_data = text_data[split_idx:]

        # DataLoaders
        self.train_loader = self.create_dataloader_v1(
            train_data,
            batch_size=16,
            context_length=256,
            stride=256,
            drop_last=True,
            shuffle=True,
            num_workers=0,
        )
        self.val_loader = self.create_dataloader_v1(
            val_data,
            batch_size=16,
            context_length=256,
            stride=256,
            drop_last=False,
            shuffle=False,
            num_workers=0,
        )

        # Model/device/optimizer
        self.model = self.createGPTModel(self.MODEL_CONFIG)
        self.device = self.torch.device(
            "cuda" if self.torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        total_size_mb = (total_params * 4) / (1024 * 1024)
        print(f"Total number of parameters: {total_params:,}")
        print(f"Total size of the model: {total_size_mb:.2f} MB")

        self.optimizer = self.torch.optim.AdamW(
            self.model.parameters(), lr=4e-4, weight_decay=0.1
        )

    # ---------------- Helpers ----------------
    def calc_loss_batch(self, input_batch, target_batch):
        torch = self.torch
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), target_batch.flatten()
        )
        return loss

    def calc_loss_loader(self, data_loader, num_batches=None):
        torch = self.torch
        total_loss = 0.0
        if len(data_loader) == 0:
            return float("nan")

        if num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        self.model.eval()
        with torch.no_grad():
            for i, (input_batch, target_batch) in enumerate(data_loader):
                if i >= num_batches:
                    break
                loss = self.calc_loss_batch(input_batch, target_batch)
                total_loss += loss.item()
        self.model.train()
        return total_loss / num_batches

    def generate_text(self, idx, max_new_tokens, context_size):
        torch = self.torch
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = self.model(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def text_to_token_ids(self, text):
        torch = self.torch
        encoded = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        encoded_tensor = torch.tensor(encoded, device=self.device).unsqueeze(0)
        return encoded_tensor

    def token_ids_to_text(self, token_ids):
        flat = token_ids.squeeze(0).tolist()
        return self.tokenizer.decode(flat)

    def generate_and_print_sample(self, start_context, max_new_tokens=50):
        context_size = self.model.pos_emb.weight.shape[0]
        encoded = self.text_to_token_ids(start_context)
        with self.torch.no_grad():
            token_ids = self.generate_text(
                idx=encoded, max_new_tokens=max_new_tokens, context_size=context_size
            )
        decoded_text = self.token_ids_to_text(token_ids)
        print(decoded_text.replace("\n", " "))

    # ----------------- Checkpointing -----------------
    def save_checkpoint(self, base_name: str = "model_and_optimizer.pth"):
        # import time, os

        run_id = time.strftime("%Y%m%d-%H%M%S")  
        rel_dir = f"runs/{run_id}"
        mount_root = "/checkpoints"              
        abs_dir = os.path.join(mount_root, rel_dir)
        os.makedirs(abs_dir, exist_ok=True)

        abs_path = os.path.join(abs_dir, base_name)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.MODEL_CONFIG,
        }
        self.torch.save(payload, abs_path)

        CHECKPOINTS_VOL.commit()

        size_mb = os.path.getsize(abs_path) / (1024 * 1024)
        print(f"Saved checkpoint to {abs_path} ({size_mb:.1f} MB)")
        return {
            "volume_rel_path": f"{rel_dir}/{base_name}",  # path relative to volume root
            "run_id": run_id,
            "size_mb": size_mb,
        }

    # ---------------- Train ----------------
    @modal.method()
    def train(
        self,
        num_epochs: int = 10,
        eval_freq: int = 5,
        eval_iter: int = 5,
        start_context: str = "The verdict of the court is as follows: ",
        save_name: str = "model_and_optimizer.pth",
    ):
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1
        total_batches = len(self.train_loader)

        for epoch in range(num_epochs):
            self.model.train()
            epoch_start = time.time()
            for batch_idx, (input_batch, target_batch) in enumerate(
                self.train_loader, start=1
            ):
                self.optimizer.zero_grad()
                loss = self.calc_loss_batch(input_batch, target_batch)
                loss.backward()
                self.optimizer.step()

                tokens_seen += input_batch.numel()  # torch.numel() is a function that returns the total number of elements in a given tensor. It calculates the product of all dimensions of the tensor, effectively counting every individual value stored within it
                global_step += 1

                if global_step % eval_freq == 0:
                    train_loss = self.calc_loss_loader(
                        self.train_loader, num_batches=eval_iter
                    )
                    val_loss = self.calc_loss_loader(
                        self.val_loader, num_batches=eval_iter
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    progress = 100 * batch_idx / total_batches
                    elapsed = time.time() - epoch_start
                    est_total = elapsed / (batch_idx / total_batches + 1e-9)
                    eta = est_total - elapsed
                    print(
                        f"[Epoch {epoch + 1}/{num_epochs}] "
                        f"Batch {batch_idx:04d}/{total_batches} ({progress:5.1f}%) | "
                        f"Step {global_step:06d} | "
                        f"Train: {train_loss:.3f} | Val: {val_loss:.3f} | "
                        f"Tokens: {tokens_seen / 1e6:.2f}M | "
                        f"ETA: {eta / 60:.1f} min"
                    )

            # Sample after each epoch
            self.generate_and_print_sample(start_context)

        print("Training complete. Saving checkpoint...")
        checkpoint_info = self.save_checkpoint(base_name=save_name)
        print("Checkpoint info:", checkpoint_info)
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "tokens_seen": track_tokens_seen,
            "checkpoint_info": checkpoint_info,
        }


# ---------- Local entrypoint to run this ----------
@app.local_entrypoint()
def main(
    num_epochs: int = 1,
    eval_freq: int = 5,
    eval_iter: int = 5,
    start_context: str = "The verdict of the court is ",
    local_out: str = "./checkpoints/model_and_optimizer.pth",
):
    trainer = TrainingScript()
    result = trainer.train.remote(
        num_epochs=num_epochs,
        eval_freq=eval_freq,
        eval_iter=eval_iter,
        start_context=start_context,
        save_name=os.path.basename(local_out),
    )

    rel_path = result["checkpoint_info"]["volume_rel_path"]  
    os.makedirs(os.path.dirname(local_out) or ".", exist_ok=True)


    try:
        with open(local_out, "wb") as f:
            for chunk in CHECKPOINTS_VOL.read_file(rel_path):
                f.write(chunk)
    except FileNotFoundError as e:
        print(f"Could not find {rel_path} in volume. Listing volume to help debug...")
        entries = CHECKPOINTS_VOL.listdir("/", recursive=True)
        for ent in entries:
            print("-", ent.path)
        raise

    # print("Training finished. Metrics:", {k: v for k, v in result.items() if k != "checkpoint"})
    print(f"Checkpoint downloaded to: {os.path.abspath(local_out)}")



# modal run training.py --num-epochs 1 --eval-freq 5 --eval-iter 5 --start-context "The verdict of the court is " --local-out ./checkpoints/model_and_optimizer.pth

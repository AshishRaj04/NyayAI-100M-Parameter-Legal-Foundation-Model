import os
import math
import time
import json
import modal

app = modal.App("LLM_Training")

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("llm_engine")
    .add_local_python_source("data_loader")
    .add_local_file("./data/combined_legal_data.txt", remote_path="/app/combined_legal_data.txt")
)

CHECKPOINTS_VOL = modal.Volume.from_name("llm_checkpoints", create_if_missing=True)


@app.cls(
    image=image,
    gpu="A100-40GB",
    secrets=[modal.Secret.from_name("llm_training_secret")],
    volumes={"/checkpoints": CHECKPOINTS_VOL},
    timeout=36000 
)
class TrainingScript:
    @modal.enter()
    def setup(self):
        import torch
        import tiktoken

        from data_loader import create_dataloader_v1, tokenize_file_chunked
        from llm_engine import createGPTModel

        self.torch = torch
        self.tiktoken = tiktoken
        self.create_dataloader_v1 = create_dataloader_v1
        self.createGPTModel = createGPTModel

        # Model config
        self.MODEL_CONFIG = {
            "vocab_size": 50257,
            "context_length": 512,
            "emb_dim": 768,
            "n_heads": 12,
            "n_layers": 9,
            "drop_rate": 0.1,
            "kqv_bias": False,
        }

        # Training hyperparameters
        self.peak_lr = 4e-4
        self.min_lr = 4e-5
        self.warmup_steps = 2000

        # Tokenizer
        self.tokenizer = self.tiktoken.get_encoding("gpt2")

        # Load + tokenize data in chunks 
        file_path = "/app/combined_legal_data.txt"
        print("Tokenizing dataset in chunks...")
        all_token_ids, total_tokens = tokenize_file_chunked(file_path)
        print(f"Total tokens: {total_tokens:,}")

        # Train/val split on token tensor
        train_ratio = 0.9
        split_idx = int(train_ratio * len(all_token_ids))
        train_token_ids = all_token_ids[:split_idx]
        val_token_ids = all_token_ids[split_idx:]
        print(f"Train tokens: {len(train_token_ids):,} | Val tokens: {len(val_token_ids):,}")

        # DataLoaders 
        self.train_loader = self.create_dataloader_v1(
            train_token_ids,
            batch_size=16,
            context_length=512,
            stride=512,
            drop_last=True,
            shuffle=True,
            num_workers=0,
        )
        self.val_loader = self.create_dataloader_v1(
            val_token_ids,
            batch_size=16,
            context_length=512,
            stride=512,
            drop_last=False,
            shuffle=False,
            num_workers=0,
        )
        print(f"Train batches: {len(self.train_loader):,} | Val batches: {len(self.val_loader):,}")

        del all_token_ids, train_token_ids, val_token_ids

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
            self.model.parameters(), lr=self.peak_lr, weight_decay=0.1
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

    def generate_text(self, idx, max_new_tokens, context_size,
                      temperature=0.8, top_k=40):
        """Generate text with temperature scaling and top-k sampling."""
        torch = self.torch
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = self.model(idx_cond)
            logits = logits[:, -1, :]  # (B, V)

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
    def save_checkpoint(self, base_name: str = "model_and_optimizer.pth", extra_data: dict = None):
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
        if extra_data:
            payload.update(extra_data)

        self.torch.save(payload, abs_path)

        CHECKPOINTS_VOL.commit()

        size_mb = os.path.getsize(abs_path) / (1024 * 1024)
        print(f"Saved checkpoint to {abs_path} ({size_mb:.1f} MB)")
        return {
            "volume_rel_path": f"{rel_dir}/{base_name}",  # path relative to volume root
            "run_id": run_id,
            "size_mb": size_mb,
        }

    # ---------------- LR Schedule ----------------
    def get_lr(self, step, total_steps):
        """Cosine learning rate schedule with linear warmup."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.peak_lr * (step + 1) / self.warmup_steps
        # Cosine decay
        progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
            1 + math.cos(math.pi * progress)
        )

    # ---------------- Train ----------------
    @modal.method()
    def train(
        self,
        num_epochs: int = 10,
        eval_freq: int = 5,
        eval_iter: int = 5,
        start_context: str = "The verdict of the court is as follows: ",
        save_name: str = "model_and_optimizer.pth",
        resume_path: str = None,  # Path relative to volume root, e.g. "runs/X/epoch_1.pth"
    ):
        train_losses, val_losses, track_tokens_seen = [], [], []
        track_global_steps, track_lrs = [], []
        epoch_times = []
        tokens_seen, global_step = 0, -1
        total_batches = len(self.train_loader)
        total_steps = total_batches * num_epochs
        training_start = time.time()
        start_epoch = 0

        # --- Resumption Logic ---
        if resume_path:
            print(f"Attempting to resume from: {resume_path}")
            # Check if path exists in volume mount
            full_path = os.path.join("/checkpoints", resume_path)
            if os.path.exists(full_path):
                print(f"Loading checkpoint from {full_path}...")
                checkpoint = self.torch.load(full_path)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                
                if "epoch" in checkpoint:
                    start_epoch = checkpoint["epoch"] + 1
                    print(f"Resumed successfully! Starting from epoch {start_epoch + 1}")
                else:
                    print("Warning: Checkpoint did not contain 'epoch'. Starting from epoch 1.")
            else:
                print(f"Error: Checkpoint {resume_path} not found in volume! Starting from scratch.")

        print(f"Total training steps: {total_steps:,} ({total_batches:,} batches x {num_epochs} epochs)")
        print(f"LR schedule: warmup {self.warmup_steps} steps -> cosine decay {self.peak_lr} -> {self.min_lr}")

        for epoch in range(start_epoch, num_epochs):
            self.model.train()
            epoch_start = time.time()
            epoch_loss_sum = 0.0
            epoch_batches = 0

            for batch_idx, (input_batch, target_batch) in enumerate(
                self.train_loader, start=1
            ):
                # Update learning rate
                global_step += 1
                lr = self.get_lr(global_step, total_steps)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                self.optimizer.zero_grad()
                loss = self.calc_loss_batch(input_batch, target_batch)
                loss.backward()

                # Gradient clipping
                self.torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )

                self.optimizer.step()

                tokens_seen += input_batch.numel()
                epoch_loss_sum += loss.item()
                epoch_batches += 1

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
                    track_global_steps.append(global_step)
                    track_lrs.append(lr)
                    progress = 100 * batch_idx / total_batches
                    elapsed = time.time() - epoch_start
                    est_total = elapsed / (batch_idx / total_batches + 1e-9)
                    eta = est_total - elapsed
                    print(
                        f"[Epoch {epoch + 1}/{num_epochs}] "
                        f"Batch {batch_idx:04d}/{total_batches} ({progress:5.1f}%) | "
                        f"Step {global_step:06d} | "
                        f"Train: {train_loss:.3f} | Val: {val_loss:.3f} | "
                        f"LR: {lr:.2e} | "
                        f"Tokens: {tokens_seen / 1e6:.2f}M | "
                        f"ETA: {eta / 60:.1f} min"
                    )

            # Epoch stats
            epoch_elapsed = time.time() - epoch_start
            avg_epoch_loss = epoch_loss_sum / max(epoch_batches, 1)
            epoch_times.append(epoch_elapsed)
            print(
                f"--- Epoch {epoch + 1} done in {epoch_elapsed / 60:.1f} min | "
                f"Avg batch loss: {avg_epoch_loss:.4f} ---"
            )

            # Sample after each epoch
            self.generate_and_print_sample(start_context)

            # Save per-epoch checkpoint
            epoch_ckpt_name = f"epoch_{epoch + 1}_{save_name}"
            print(f"Saving epoch {epoch + 1} checkpoint...")
            checkpoint_info = self.save_checkpoint(
                base_name=epoch_ckpt_name, 
                extra_data={"epoch": epoch}
            )

            # Save per-epoch training log to volume (safety net)
            epoch_log = {
                "model_config": self.MODEL_CONFIG,
                "completed_epochs": epoch + 1,
                "total_epochs": num_epochs,
                "metrics": {
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "tokens_seen": track_tokens_seen,
                    "global_steps": track_global_steps,
                    "learning_rates": track_lrs,
                },
                "epoch_times_sec": epoch_times,
                "total_tokens_seen": tokens_seen,
                "checkpoint_info": checkpoint_info,
            }
            log_path = os.path.join(
                "/checkpoints",
                f"runs/{checkpoint_info['run_id']}",
                "training_log.json",
            )
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w") as f:
                json.dump(epoch_log, f, indent=2)
            CHECKPOINTS_VOL.commit()
            print(f"Epoch {epoch + 1} training log saved to {log_path}")
            
            # Yield stats for this epoch so client can download immediately
            epoch_data = {
                "type": "epoch_complete",
                "epoch": epoch + 1,
                "train_loss": train_losses[-1] if train_losses else None,
                "val_loss": val_losses[-1] if val_losses else None,
                "checkpoint_info": checkpoint_info,
                "log": epoch_log,
            }
            yield epoch_data

        total_training_time = time.time() - training_start

        print("Training complete. Saving final checkpoint...")
        checkpoint_info = self.save_checkpoint(
            base_name=save_name,
            extra_data={"epoch": num_epochs - 1}
        )
        print("Checkpoint info:", checkpoint_info)

        # Build training log
        training_log = {
            "model_config": self.MODEL_CONFIG,
            "training_params": {
                "num_epochs": num_epochs,
                "eval_freq": eval_freq,
                "eval_iter": eval_iter,
                "batch_size": 16,
                "peak_lr": self.peak_lr,
                "min_lr": self.min_lr,
                "warmup_steps": self.warmup_steps,
                "weight_decay": 0.1,
                "grad_clip_norm": 1.0,
                "context_length": 512,
                "stride": 512,
            },
            "metrics": {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "tokens_seen": track_tokens_seen,
                "global_steps": track_global_steps,
                "learning_rates": track_lrs,
            },
            "epoch_times_sec": epoch_times,
            "total_training_time_sec": total_training_time,
            "total_tokens_seen": tokens_seen,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "checkpoint_info": checkpoint_info,
        }

        # Save log to volume
        log_path = os.path.join(
            "/checkpoints",
            f"runs/{checkpoint_info['run_id']}",
            "training_log.json",
        )
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)
        CHECKPOINTS_VOL.commit()
        print(f"Training log saved to {log_path}")

        yield {
             "type": "training_complete",
             "log": training_log
        }


# ---------- Local entrypoint to run this ----------
@app.local_entrypoint()
def main(
    num_epochs: int = 1,
    eval_freq: int = 5,
    eval_iter: int = 5,
    start_context: str = "The verdict of the court is ",
    local_out: str = "./checkpoints/model_and_optimizer.pth",
    resume_from: str = None,  # Optional path to resume, e.g. "runs/2026.../epoch_2..."
):
    trainer = TrainingScript()
    
    # Check output dir
    os.makedirs(os.path.dirname(local_out) or ".", exist_ok=True)
    
    print("Starting training...")
    if resume_from:
        print(f"Requesting resume from: {resume_from}")

    # Iterate over the generator to process results per epoch
    final_log = None
    
    for result in trainer.train.remote_gen(
        num_epochs=num_epochs,
        eval_freq=eval_freq,
        eval_iter=eval_iter,
        start_context=start_context,
        save_name=os.path.basename(local_out),
        resume_path=resume_from,
    ):
        result_type = result.get("type", "unknown")
        
        if result_type == "epoch_complete":
            epoch = result["epoch"]
            ckpt_info = result["checkpoint_info"]
            rel_path = ckpt_info["volume_rel_path"]
            
            print(f"\n>> Epoch {epoch} complete!")
            print(f"   Downloading checkpoint: {rel_path}")
            
            # Construct local filename: epoch_N_model_and_optimizer.pth
            local_name = os.path.basename(rel_path)
            local_epoch_path = os.path.join(os.path.dirname(local_out), local_name)
            
            try:
                with open(local_epoch_path, "wb") as f:
                    for chunk in CHECKPOINTS_VOL.read_file(rel_path):
                        f.write(chunk)
                print(f"   Saved locally to: {local_epoch_path}")
            except Exception as e:
                print(f"   Failed to download checkpoint: {e}")

            # Save per-epoch training log locally
            epoch_log = result.get("log")
            if epoch_log:
                epoch_log_path = os.path.join(
                    os.path.dirname(local_out) or ".",
                    f"training_log_epoch_{epoch}.json"
                )
                with open(epoch_log_path, "w") as f:
                    json.dump(epoch_log, f, indent=2)
                print(f"   Training log saved to: {epoch_log_path}")

        elif result_type == "training_complete":
            final_log = result["log"]

    # Download final checkpoint (should be same as last epoch, but good to have the canonical name)
    if final_log:
        rel_path = final_log["checkpoint_info"]["volume_rel_path"]
        run_id = final_log["checkpoint_info"]["run_id"]
        
        print("\n>> Downloading FINAL checkpoint...")
        try:
            with open(local_out, "wb") as f:
                for chunk in CHECKPOINTS_VOL.read_file(rel_path):
                    f.write(chunk)
            print(f"Saved locally to: {os.path.abspath(local_out)}")
        except Exception as e:
            print(f"Failed to download final checkpoint: {e}")

        # Download training log
        log_rel_path = f"runs/{run_id}/training_log.json"
        local_log_path = os.path.join(
            os.path.dirname(local_out) or ".", "training_log.json"
        )
        try:
            with open(local_log_path, "wb") as f:
                for chunk in CHECKPOINTS_VOL.read_file(log_rel_path):
                    f.write(chunk)
            print(f"Training log downloaded to: {os.path.abspath(local_log_path)}")
        except Exception as e:
            # Fallback
            with open(local_log_path, "w") as f:
                json.dump(final_log, f, indent=2)
            print(f"Training log saved from result to: {os.path.abspath(local_log_path)}")

        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        total_time = final_log.get("total_training_time_sec", 0)
        print(f"  Total time    : {total_time / 60:.1f} minutes")
        print(f"  Tokens seen   : {final_log.get('total_tokens_seen', 0) / 1e6:.1f}M")
        print(f"  Final train loss: {final_log.get('final_train_loss', 'N/A')}")
        print(f"  Final val loss  : {final_log.get('final_val_loss', 'N/A')}")
        print(f"  Log file      : {os.path.abspath(local_log_path)}")
        print("=" * 60)


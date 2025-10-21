"""Module for training the model with contrastive loss.

This module defines the training loop for a model that learns to embed vocal and instrumental
audio segments. This module can be used for training different architectures."""

import torch
import pickle
import torch.nn.functional as F
import torchaudio.transforms as AT

def contrastive_loss(v_emb, i_emb, temperature=0.07):
    """
    InfoNCE-style contrastive loss using cosine similarity between normalized embeddings.
    """
    # Normalize embeddings
    v_emb = F.normalize(v_emb, dim=-1)
    i_emb = F.normalize(i_emb, dim=-1)

    # Cosine similarity matrix scaled by temperature
    sim_matrix = torch.matmul(v_emb, i_emb.T) / temperature  # [B, B]

    # Targets: diagonal elements are positives
    targets = torch.arange(v_emb.size(0), device=v_emb.device)

    # Cross-entropy loss in both directions
    loss_v2i = F.cross_entropy(sim_matrix, targets)
    loss_i2v = F.cross_entropy(sim_matrix.T, targets)

    return (loss_v2i + loss_i2v) / 2

def train_model(
    model, 
    train_loader, 
    val_loader, 
    epochs=50, 
    lr=1e-4, 
    weight_decay=1e-5,
    save_path="dual_encoder_best.pth", 
    device=None,
    early_stopping_patience=5,
    use_spec_augment=False
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🚀 Using device: {device}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    early_stop_counter = 0

    # Optional spec augmentation
    if use_spec_augment:
        freq_mask = AT.FrequencyMasking(freq_mask_param=15)
        time_mask = AT.TimeMasking(time_mask_param=20)
        print("🎨 Using SpecAugment for data augmentation")

    print(f"🚀 Starting training for {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        # -------------------- TRAIN --------------------
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            # Load tensors and move to GPU safely
            vocals = batch["vocals"].float()
            non_vocals = batch["non_vocals"].float()

            if use_spec_augment:
                vocals = freq_mask(vocals)
                vocals = time_mask(vocals)
                non_vocals = freq_mask(non_vocals)
                non_vocals = time_mask(non_vocals)

            vocals = vocals.to(device)
            non_vocals = non_vocals.to(device)

            # Forward pass
            v_emb, i_emb = model(vocals, non_vocals)
            loss = contrastive_loss(v_emb, i_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"[Train] Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        print(f"[Epoch {epoch}] Avg train loss: {avg_train_loss:.4f}")

        # -------------------- VALIDATE --------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                vocals = batch["vocals"].float().to(device)
                non_vocals = batch["non_vocals"].float().to(device)
                v_emb, i_emb = model(vocals, non_vocals)
                loss = contrastive_loss(v_emb, i_emb)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"[Epoch {epoch}] Avg val loss: {avg_val_loss:.4f}")

        # Scheduler step
        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        # Save best model & handle early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved new best model to {save_path}")
        else:
            early_stop_counter += 1
            print(f"⚠️ No improvement for {early_stop_counter} epochs")
            if early_stop_counter >= early_stopping_patience:
                print(f"🛑 Early stopping triggered after {early_stop_counter} epochs without improvement")
                break

    print("\n🎯 Training complete!")
    print(f"Lowest validation loss: {best_val_loss:.4f}")
    return history


def save_embeddings(model, dataloader, out_path_prefix="test", device=None):
    """
    Computes and saves all vocal and instrumental embeddings and chunk names from a dataloader.
    Args:
        model: Trained DualEncoderModel.
        dataloader: DataLoader yielding batches with 'vocals', 'non_vocals', 'chunk_name'.
        out_path_prefix: Prefix for output files (e.g., 'test' -> test_vocal_embs.pt, test_instr_embs.pt, test_names.pkl).
        device: Device to run model on.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    vocal_embs = []
    instr_embs = []
    chunk_names = []

    with torch.no_grad():
        for batch in dataloader:
            vocals = batch["vocals"].float().to(device)
            non_vocals = batch["non_vocals"].float().to(device)
            v_emb, i_emb = model(vocals, non_vocals)
            vocal_embs.append(v_emb.cpu())
            instr_embs.append(i_emb.cpu())
            chunk_names.extend(batch["chunk_name"])

    vocal_embs = torch.cat(vocal_embs, dim=0)
    instr_embs = torch.cat(instr_embs, dim=0)

    torch.save(vocal_embs, f"{out_path_prefix}_vocal_embs.pt")
    torch.save(instr_embs, f"{out_path_prefix}_instr_embs.pt")
    with open(f"{out_path_prefix}_names.pkl", "wb") as f:
        pickle.dump(chunk_names, f)

    print(f"Saved {len(chunk_names)} embeddings to {out_path_prefix}_vocal_embs.pt, {out_path_prefix}_instr_embs.pt, and {out_path_prefix}_names.pkl")

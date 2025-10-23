import pickle
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig
import torch
from tqdm import tqdm

from deepmash.data_processing.common import get_dataloaders, StemsDataset

def save_embeddings(model, loaders, output_prefix: str):
    print("Saving embeddings...")
    model.eval()
    all_vocal_embeddings = []
    all_non_vocal_embeddings = []
    all_names = []
    for loader in tqdm(loaders):
        for batch in tqdm(loader):
            vocal_embs, non_vocal_embs, names = model.compute_embeddings(batch)
            all_vocal_embeddings.append(vocal_embs.cpu())
            all_non_vocal_embeddings.append(non_vocal_embs.cpu())
            all_names.extend(names)
    all_vocal_embeddings_tensor = torch.cat(all_vocal_embeddings, dim=0)  # [N, D]
    all_non_vocal_embeddings_tensor = torch.cat(all_non_vocal_embeddings, dim=0)  # [N, D]
    emb_file = f"{output_prefix}_vocal_embs.pt"
    non_emb_file = f"{output_prefix}_non_vocal_embs.pt"
    names_file = f"{output_prefix}_names.pkl"
    torch.save(all_vocal_embeddings_tensor, emb_file)
    torch.save(all_non_vocal_embeddings_tensor, non_emb_file)
    with open(names_file, "wb") as f:
        pickle.dump(all_names, f)
    print(f"Saved embeddings to {emb_file}, {non_emb_file} and names to {names_file}")


def training_run(
    dataset: StemsDataset,
    model: L.LightningModule,
    config: DictConfig,
    log_every_n_steps: int = 10
):
    
    # TODO: experiment with batch size and num_workers)
    # seems like increasing num_workers just makes it slower for me atleast
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=dataset,
        config=config.dataset,
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",          # metric to monitor
        mode="min",                  # or "max" for accuracy
        save_top_k=1,                # keep only the best checkpoint
        filename="best-checkpoint-{epoch:02d}",
        verbose=False
    )

    logger = CSVLogger(save_dir=config.training.logger_dir, name=config.training.log_name or model.__class__.__name__)

    trainer = L.Trainer(
        max_epochs=config.training.max_epochs,
        overfit_batches=config.training.overfit_batches,
        accelerator="auto",
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=True,
        val_check_interval=1.0,     # check val once every epoch
        callbacks=[checkpoint_callback],
    ) 

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")

    print("Training complete.")
    # save embeddings 
    if config.training.save_embeddings:
        save_embeddings(model=model, loaders=[train_loader, val_loader, test_loader], output_prefix=config.data.save_model)
    
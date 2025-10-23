import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig

from deepmash.data_processing.common import get_dataloaders, StemsDataset

def training_run(
    dataset: StemsDataset,
    model: L.LightningModule,
    config: DictConfig,
    log_every_n_steps: int = 10,
    checkpoint_path: str = None  # Optional parameter for checkpoint path
):
    
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=dataset,
        config=config.dataset,
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_top_1",          # metric to monitor
        mode="max",                  # or "max" for accuracy
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

    trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)
    trainer.test(model, test_loader, ckpt_path="best")

def test_run(
    dataset: StemsDataset,
    model: L.LightningModule,
    config: DictConfig,
    checkpoint_path: str = "best"  # Default to "best" checkpoint
):
    # Load the test dataset
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=dataset,
        config=config.dataset,
    )
    
    # Load the specified model checkpoint
    trainer = L.Trainer(
        accelerator="auto",
        enable_progress_bar=True,
    )

    # Test the model with the specified checkpoint
    trainer.test(model, test_loader, ckpt_path=checkpoint_path)

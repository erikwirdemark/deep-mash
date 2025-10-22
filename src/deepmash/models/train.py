import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig
from time import perf_counter

from deepmash.data_processing.common import get_dataloaders, StemsDataset

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
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=config.training.early_stopping_patience,
        min_delta=0,
    )


    logger = CSVLogger(save_dir=config.training.logger_dir, name=config.training.log_name or model.__class__.__name__)
    
    t0 = perf_counter()

    trainer = L.Trainer(
        max_epochs=config.training.max_epochs,
        overfit_batches=config.training.overfit_batches,
        accelerator="auto",
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=True,
        val_check_interval=1.0,     # check val once every epoch
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"Training completed in {perf_counter() - t0:.2f} seconds.")
    
    trainer.test(model, test_loader, ckpt_path="best")
    
    return trainer
    
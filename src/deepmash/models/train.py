import lightning as L
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig

from deepmash.data_processing.common import get_dataloaders, StemsDataset

def training_run(
    dataset: StemsDataset,
    model: L.LightningModule,
    config: DictConfig,
    log_every_n_steps: int = 10,
    val_split: float = 0.1,
    test_split: float = 0.1
):
    
    # TODO: experiment with batch size and num_workers)
    # seems like increasing num_workers just makes it slower for me atleast
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_split=val_split,
        test_split=test_split
    )

    logger = CSVLogger(save_dir=config.logger_dir, name=config.log_name or model.__class__.__name__)

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        overfit_batches=config.overfit_batches,
        accelerator="auto",
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=True,
        val_check_interval=1.0,     # check val once every epoch
        enable_checkpointing=False, # for now, but should enable later
    ) 

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    
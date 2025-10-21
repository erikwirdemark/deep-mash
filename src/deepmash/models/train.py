import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from deepmash.data_processing.common import get_dataloaders, StemsDataset

def training_run(
    dataset: StemsDataset,
    model: L.LightningModule,
    max_epochs: int = 10, 
    batch_size: int = 16, 
    num_workers: int = 0,
    overfit_batches: float = 0.0,
    
    logger_dir: str = "../logs",
    log_every_n_steps: int = 10,
    log_name: str|None = None,
):
    
    # TODO: experiment with batch size and num_workers)
    # seems like increasing num_workers just makes it slower for me atleast
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=0.1,
        test_split=0.1
    )
    
    logger = CSVLogger(save_dir=logger_dir, name=log_name or model.__class__.__name__)
    
    trainer = L.Trainer(
        max_epochs=max_epochs,
        overfit_batches=overfit_batches,
        accelerator="auto",
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=True,
        val_check_interval=1.0,     # check val once every epoch
        enable_checkpointing=False, # for now, but should enable later
    ) 

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    

    
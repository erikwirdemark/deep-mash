from omegaconf import DictConfig
from deepmash.data_processing.common import ToLogMel


def get_dataloaders(config: DictConfig, preprocess: bool=False, num_workers: int=0):
    if config.dataset.name == "musdb18":
        from deepmash.data_processing.musdb18 import MUSDB18StemsDataset
        dataset = MUSDB18StemsDataset(
            config=config,
            split=config.data.split,
            preprocess=preprocess,
            preprocess_transform=None
        )
    elif config.dataset.name == "gtzan_stems":
        from deepmash.data_processing.gtzan_stems import GTZANStemsDataset
        dataset = GTZANStemsDataset(
            config=config,
            already_preprocessed=not preprocess, # Hur ska denna användas, ska vi ha en check?
            preprocess=preprocess,
            preprocess_transform=ToLogMel(config=config)
        )
    return get_dataloaders(dataset=dataset, config=config.dataset)

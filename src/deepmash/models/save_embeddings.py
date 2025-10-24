from omegaconf import OmegaConf
from deepmash.data_processing.common import ToLogMel, get_dataloaders
from deepmash.data_processing.gtzan_stems import GTZANStemsDataset
from deepmash.models.cnn import CNN
from deepmash.models.train import save_embeddings

config = OmegaConf.load('/Users/erikwirdemark/deep-mash/config/query_default.yaml')
model = CNN.load_from_checkpoint('/Users/erikwirdemark/deep-mash/best-checkpoint-epoch=17.ckpt', config=config.model)
dataset = dataset = GTZANStemsDataset(
            config=config,
            already_preprocessed=True, # Hur ska denna användas, ska vi ha en check?
            preprocess_transform=ToLogMel(config=config)
        )
loaders, _, _ = get_dataloaders(
        dataset=dataset,
        config=config.dataset,
    )
save_embeddings(model=model, loaders=[loaders], output_prefix='/Users/erikwirdemark/deep-mash/data/saved_models/baseline')
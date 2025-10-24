import argparse
import sys
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf
from deepmash import CocolaCNN, CNN, training_run, MUSDB18Dataset, GTZANStemsDataset, ToLogMel, query_saved_embeddings, mashup

#!/usr/bin/env python3
"""
CLI for preprocessing, modeling, querying and mixing given a config file.

"""

def train(config: DictConfig):
    # Load dataloaders
    if config.dataset.name == "musdb18":
        dataset = MUSDB18Dataset(
            config=config,
            split=config.data.split,
            already_preprocessed=True, # Hur ska denna användas, ska vi ha en check?
            preprocess_transform=None
        )
    elif config.dataset.name == "gtzan_stems":
        dataset = GTZANStemsDataset(
            config=config,
            already_preprocessed=True, # Hur ska denna användas, ska vi ha en check?
            preprocess_transform=ToLogMel(config=config)
        )
    if config.model.name == "cocola_cnn":
        model = CocolaCNN(config=config.model)
    elif config.model.name == "cnn":
        model = CNN(config=config.model)
    else:
        raise ValueError(f"Unknown model name: {config.model.name}")
    # Train model
    training_run(dataset=dataset, model=model, config=config)


def query_model(config: DictConfig, query: Optional[str]) -> int:
    model_path = config.query.model_path
    catalogue_path = config.query.catalogue_path
    model = CNN.load_from_checkpoint(model_path) if config.model.name == "cnn" else CocolaCNN.load_from_checkpoint(model_path)
    model.eval()
    results = query_saved_embeddings(
        model=model,
        query_audio=query,
        catalogue=catalogue_path,
        target_sr=config.audio.target_sample_rate,
        top_k=config.query.top_k,
        preprocess_transform=ToLogMel(config=config)
        )
    index = input("Select track to mix with...\n")
    print(f"Mixing your song...")
    name = results[int(index)-1][0]
    genre = name.split(".")[0]
    track = '.'.join(name.split(".")[:-1])
    print(f"Selected track: {name}")
    non_vocals_path = '/Users/erikwirdemark/deep-mash/data/genres_stems/' + genre + '/' + track 
    mashup(config=config, vocals_path=Path(query), track_folder=Path(non_vocals_path), plot=False)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="deep-mash", description="Preprocess / Model / Query / mix CLI")
    p.add_argument("--config", "-c", type=Path, required=True, help="Path to JSON/YAML config file")
    p.add_argument("--dry-run", action="store_true", help="Print actions without executing them")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="Run training step")

    q_parser = sub.add_parser("query", help="Run a query against the model/service")
    q_parser.add_argument("--q", "-q", type=str, help="Query string")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    config = OmegaConf.load(args.config)

    if args.command == "train":
        return train(config=config)
    if args.command == "query":
        return query_model(config, args.q)

    parser.print_help()
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
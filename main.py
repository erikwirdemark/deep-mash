import argparse
import sys
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf
from deep_mash import train_model, DualEncoderModel, create_dataloaders, save_embeddings

#!/usr/bin/env python3
"""
CLI for preprocessing, modeling, querying and mixing given a config file.

"""

def preprocess(config: DictConfig, dry_run: bool = False) -> int:
    pass


def train(config: DictConfig):
    # Load dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(preprocess=True, config=config)
    model = DualEncoderModel(config=config)
    # Train model
    save_path = config.data.save_model + "/baseline.pt"
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        save_path=save_path
        )
    save_embeddings(model=model, dataloader=test_loader, out_path_prefix=config.data.train_embeddings_prefix)


def query_model(config: DictConfig, query: Optional[str], dry_run: bool = False) -> int:
    pass


def mix_action(config: DictConfig, direction: str, dry_run: bool = False) -> int:
    pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="deep-mash", description="Preprocess / Model / Query / mix CLI")
    p.add_argument("--config", "-c", type=Path, required=True, help="Path to JSON/YAML config file")
    p.add_argument("--dry-run", action="store_true", help="Print actions without executing them")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("preprocess", help="Run preprocessing step")
    sub.add_parser("train", help="Run training step")

    q_parser = sub.add_parser("query", help="Run a query against the model/service")
    q_parser.add_argument("--q", "-q", type=str, help="Query string (overrides config default)")

    s_parser = sub.add_parser("mix", help="mix artifacts (push/pull)")
    s_parser.add_argument("direction", choices=["push", "pull"], help="mix direction")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    config = OmegaConf.load(args.config)

    if args.command == "preprocess":
        return preprocess(config, dry_run=args.dry_run)
    if args.command == "train":
        print('command sent')
        return train(config=config)
    if args.command == "query":
        return query_model(config, args.q, dry_run=args.dry_run)
    if args.command == "mix":
        return mix_action(config, args.direction, dry_run=args.dry_run)

    parser.print_help()
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
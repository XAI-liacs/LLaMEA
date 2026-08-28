"""Fine-tunes (or trains-from-scratch) the RLM surrogate on the examples
produced by ``data_pipeline.py``.

Two usage patterns (Step 2):
  1. Train-from-scratch-on-this-run: point ``--train``/``--val`` at one run's
     split, leave ``pretrained_checkpoint`` unset in the config.
  2. Pretrain-on-pool, fine-tune-on-run: first run this script over a pooled
     set of historical logs to produce a checkpoint, then run it again with
     ``configs/finetune_from_pool.yaml`` (``pretrained_checkpoint`` pointing
     at that checkpoint) and ``--train``/``--val`` pointing at the new run's
     (typically early-generation) split.

CLI:
    python -m llamea.rlm_surrogate.train \\
        --config llamea/rlm_surrogate/configs/default.yaml \\
        --train data/train.jsonl --val data/val.jsonl \\
        --output-dir checkpoints/run1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .config import RLMSurrogateConfig
from .data_pipeline import read_examples_jsonl
from .model import build_model, build_optimizer_factory, infer_max_num_objs


def _to_core_examples(examples):
    from regress_lm import core

    return [core.Example(x=e.x, y=e.y) for e in examples]


def train(
    config: RLMSurrogateConfig,
    train_path: str | Path,
    val_path: str | Path | None,
    output_dir: str | Path,
) -> dict:
    import torch

    if torch.cuda.is_available():
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(
            "WARNING: torch.cuda.is_available() is False -- training will run "
            "on CPU. If you're on a GPU machine, check `nvidia-smi` and your "
            "torch/CUDA install (see README.md)."
        )

    train_examples = read_examples_jsonl(train_path)
    if not train_examples:
        raise ValueError(f"No training examples found at {train_path}")
    val_examples = read_examples_jsonl(val_path) if val_path else None

    needed_objs = infer_max_num_objs(train_examples + (val_examples or []))
    if needed_objs > config.max_num_objs:
        config = config.with_overrides(max_num_objs=needed_objs)

    reg_lm = build_model(config)

    if config.pretrained_checkpoint:
        state = torch.load(config.pretrained_checkpoint, map_location="cpu")
        reg_lm.model.load_state_dict(state)

    effective_batch_size = config.batch_size or len(train_examples)
    updates_per_epoch = math.ceil(len(train_examples) / effective_batch_size)
    total_steps = config.total_steps or config.max_epochs * updates_per_epoch
    optimizer_factory = build_optimizer_factory(config, total_steps=total_steps)

    reg_lm.fine_tune(
        _to_core_examples(train_examples),
        validation_examples=_to_core_examples(val_examples) if val_examples else None,
        seed=config.seed,
        optimizer_factory=optimizer_factory,
        max_epochs=config.max_epochs,
        batch_size=config.batch_size,
        batch_size_per_device=config.batch_size_per_device,
        patience=config.patience,
        use_lora=config.use_lora,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(reg_lm.model.state_dict(), output_dir / "model.pt")
    config.to_yaml(output_dir / "config.yaml")

    manifest = {
        "n_train": len(train_examples),
        "n_val": len(val_examples) if val_examples else 0,
        "train_path": str(train_path),
        "val_path": str(val_path) if val_path else None,
        "total_steps": total_steps,
        "max_num_objs": config.max_num_objs,
    }
    with open(output_dir / "train_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    default_config = Path(__file__).parent / "configs" / "default.yaml"
    p.add_argument("--config", default=str(default_config))
    p.add_argument("--train", required=True)
    p.add_argument("--val", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--encoder-type", default=None, choices=["t5gemma", "vanilla", "char"]
    )
    p.add_argument("--pretrained-checkpoint", default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    config = RLMSurrogateConfig.from_yaml(args.config)
    config = config.with_overrides(
        encoder_type=args.encoder_type,
        pretrained_checkpoint=args.pretrained_checkpoint,
        max_epochs=args.max_epochs,
        lr=args.lr,
        seed=args.seed,
    )
    manifest = train(config, args.train, args.val, args.output_dir)
    print(
        f"Trained on {manifest['n_train']} examples "
        f"({manifest['total_steps']} total optimizer steps); "
        f"checkpoint written to {args.output_dir}"
    )


if __name__ == "__main__":
    main()

"""Configuration for building/training the RLM surrogate.

Kept close to the RLM paper's Appendix C defaults (Adafactor, linear
warmup + cosine decay, frozen encoder, 2 decoder layers, median-of-64
sampling for point predictions) but everything is overridable via a YAML
config file and/or CLI flags -- nothing is hardcoded in the training code.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

EncoderType = Literal["t5gemma", "vanilla", "char"]
OptimizerName = Literal["adafactor", "adamw", "muon_adamw"]


@dataclass
class RLMSurrogateConfig:
    # --- Encoder ---
    # "t5gemma": real frozen pretrained encoder, as used in the paper. Requires
    #   the `transformers`/`peft` extras and a HuggingFace token with the
    #   Gemma license accepted (`huggingface-cli login`) -- gated model.
    # "vanilla": regress-lm's small from-scratch transformer encoder (uses the
    #   public T5 SentencePiece vocab, downloaded from GCS). Not a pretrained
    #   encoder; useful when the t5gemma weights aren't reachable.
    # "char": from-scratch encoder with a fully local/offline character
    #   vocab. No network access required; intended for tests/CI and for
    #   sanity-checking the pipeline before the real data/network is ready.
    encoder_type: EncoderType = "t5gemma"
    t5gemma_model_name: str = "google/t5gemma-s-s-prefixlm"
    freeze_encoder: bool = True
    d_model: int = 512  # only used by "vanilla"/"char" (from-scratch) encoders

    # --- Decoder ---
    num_decoder_layers: int = 2
    dropout: float = 0.0
    max_input_len: int = 2048
    max_num_objs: int = 1  # 1 for scalar `fitness`, len(aucs) for the aucs target
    z_loss_coef: float | None = None

    # --- Optimizer / schedule (Appendix C) ---
    optimizer: OptimizerName = "adafactor"
    lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_steps: int = 200
    total_steps: int | None = None  # if None, train.py infers from data size

    # --- Fine-tuning loop ---
    max_epochs: int = 100
    batch_size: int | None = None
    batch_size_per_device: int | None = None
    patience: int | None = 5
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    seed: int = 0

    # --- Inference ---
    num_samples_point_pred: int = 64  # median-of-64, per Appendix C

    # --- Fine-tuning-from-a-pretrained-pool configuration ---
    pretrained_checkpoint: str | None = None  # path to a checkpoint from a
    # prior "train-from-scratch" run over a pooled set of historical LLaMEA
    # logs. When set, train.py loads these weights before fine-tuning on the
    # new run's (typically early-generation) examples instead of starting
    # from random/frozen-encoder init.

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RLMSurrogateConfig":
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in raw.items() if k in known}
        extra = {k: v for k, v in raw.items() if k not in known}
        return cls(**kwargs, extra=extra)

    def to_yaml(self, path: str | Path) -> None:
        d = dataclasses.asdict(self)
        extra = d.pop("extra", {})
        d.update(extra)
        with open(path, "w") as fh:
            yaml.safe_dump(d, fh, sort_keys=False)

    def with_overrides(self, **overrides: Any) -> "RLMSurrogateConfig":
        return dataclasses.replace(
            self, **{k: v for k, v in overrides.items() if v is not None}
        )

"""Builds a ``regress_lm.rlm.RegressLM`` (encoder-decoder RLM with digit-token
decoding and constrained decoding, per Akhauri, Song et al. 2026) from an
``RLMSurrogateConfig``, plus the Adafactor/warmup+cosine optimizer factory
from the paper's Appendix C and a median-of-N point-prediction helper.

Deliberately does NOT build a custom MSE regression head -- Table 5 of the
paper shows the digit-decoder outperforms both a normalized (0.717) and
unnormalized (0.478) regression head, versus 0.800 Spearman-rho for the
decoder.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .config import RLMSurrogateConfig
from .data_pipeline import RLMExample


def build_model(config: RLMSurrogateConfig):
    """Constructs a RegressLM per ``config.encoder_type``."""
    from regress_lm import rlm as rlm_lib
    from regress_lm import vocabs

    common_kwargs: dict[str, Any] = dict(
        max_input_len=config.max_input_len,
        max_num_objs=config.max_num_objs,
        num_decoder_layers=config.num_decoder_layers,
        dropout=config.dropout,
        z_loss_coef=config.z_loss_coef,
    )

    if config.encoder_type == "t5gemma":
        return rlm_lib.RegressLM.from_t5gemma_encoder(
            model_name=config.t5gemma_model_name,
            freeze_encoder=config.freeze_encoder,
            **common_kwargs,
        )
    elif config.encoder_type == "char":
        return rlm_lib.RegressLM.from_scratch(
            encoder_vocab=vocabs.CharacterVocab(),
            d_model=config.d_model,
            **common_kwargs,
        )
    elif config.encoder_type == "vanilla":
        return rlm_lib.RegressLM.from_scratch(d_model=config.d_model, **common_kwargs)
    raise ValueError(f"Unknown encoder_type: {config.encoder_type!r}")


def infer_max_num_objs(examples: Sequence[RLMExample]) -> int:
    """For the aucs target, examples may have variable-length y; regress-lm
    pads shorter ones automatically (see README's multi-objective example),
    but max_num_objs must be at least the longest observed sequence."""
    lengths = [len(e.y) if isinstance(e.y, list) else 1 for e in examples]
    return max(lengths) if lengths else 1


# --------------------------------------------------------------------------
# Optimizer / schedule: Adafactor + linear warmup + cosine decay (Appendix C).
# --------------------------------------------------------------------------


class _ScheduledOptimizer:
    """Wraps a torch Optimizer + LRScheduler behind the plain Optimizer
    interface (`step`, `zero_grad`, `param_groups`) that
    ``PyTorchFineTuner`` expects, stepping the scheduler once per
    optimizer step."""

    def __init__(self, base_optimizer, scheduler):
        self.base_optimizer = base_optimizer
        self.scheduler = scheduler

    def step(self, closure=None):
        self.base_optimizer.step(closure)
        self.scheduler.step()

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def state_dict(self):
        return {
            "optimizer": self.base_optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, sd):
        self.base_optimizer.load_state_dict(sd["optimizer"])
        self.scheduler.load_state_dict(sd["scheduler"])


def _linear_warmup_cosine_decay(
    step: int, warmup_steps: int, total_steps: int
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer_factory(config: RLMSurrogateConfig, total_steps: int):
    """Returns an ``optimizer_factory(named_params) -> Optimizer`` matching
    the interface ``regress_lm``'s ``PyTorchFineTuner``/``Trainer`` expect.

    ``total_steps`` must be supplied by the caller (typically ``train.py``,
    which knows the training-set size and batch size) since the schedule
    length isn't knowable at config-construction time.
    """
    import torch
    from torch.optim import lr_scheduler

    def factory(named_params):
        params = [p for _, p in named_params]

        if config.optimizer == "muon_adamw":
            from regress_lm.pytorch import optimizers as rlm_optimizers

            base = rlm_optimizers.muon_adamw(
                named_params, lr=config.lr, weight_decay=config.weight_decay
            )
        elif config.optimizer == "adafactor":
            base = torch.optim.Adafactor(
                params, lr=config.lr, weight_decay=config.weight_decay
            )
        elif config.optimizer == "adamw":
            base = torch.optim.AdamW(
                params, lr=config.lr, weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer!r}")

        if config.warmup_steps <= 0 and total_steps <= 0:
            return base

        scheduler = lr_scheduler.LambdaLR(
            base,
            lr_lambda=lambda step: _linear_warmup_cosine_decay(
                step, config.warmup_steps, max(total_steps, config.warmup_steps + 1)
            ),
        )
        return _ScheduledOptimizer(base, scheduler)

    return factory


# --------------------------------------------------------------------------
# Inference: median-of-N point predictions (Appendix C default: N=64).
# --------------------------------------------------------------------------


def median_point_predictions(
    reg_lm,
    xs: Sequence[str],
    num_samples: int | None = None,
    config: RLMSurrogateConfig | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Returns an (n, max_num_objs) array of median-of-N sampled predictions.

    ``reg_lm.sample`` decodes its *entire* input in one call, as a single
    ``len(xs) * num_samples``-sequence batch (see
    ``regress_lm.pytorch.model.PyTorchModel.decode``) -- with no chunking of
    our own, a large eval set (easily thousands of rows once
    `target="aucs_per_instance"` has exploded it, e.g. via a big
    `--max-records`) turns into an enormous single decode: it can run for
    hours or outright OOM (confirmed: a real run attempted a >300GB CPU
    allocation this way). `batch_size` chunks `xs` so each call to
    `reg_lm.sample` only ever expands to `batch_size * num_samples`
    sequences, independent of how large the full eval set is. Keep this
    conservative on modest GPUs; raise it if you have headroom.
    """
    from regress_lm import core

    if num_samples is None:
        num_samples = config.num_samples_point_pred if config else 64
    medians = []
    for start in range(0, len(xs), batch_size):
        chunk = xs[start : start + batch_size]
        inputs = [core.ExampleInput(x=x) for x in chunk]
        samples = reg_lm.sample(inputs, num_samples=num_samples)
        for s in samples:
            arr = np.asarray(s, dtype=float)
            medians.append(np.median(arr, axis=0))
    return np.array(medians)


def scalar_point_predictions(median_predictions: np.ndarray) -> np.ndarray:
    """Collapses (n, max_num_objs) median predictions to an (n,) scalar
    ranking score. For the aucs target, the mean across objectives is used
    as the fitness-like point estimate (aucs already averages to `fitness`
    in the BLADE schema)."""
    if median_predictions.ndim == 1:
        return median_predictions
    return median_predictions.mean(axis=1)

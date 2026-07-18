"""Code-to-performance surrogate model for LLaMEA, based on Regression
Language Models (RLM; Akhauri, Song et al. 2026, arXiv:2509.26476) and the
``regress-lm`` library (https://github.com/google-deepmind/regress-lm).

Submodules are intentionally kept independent so each stage can be rerun
without repeating the others:

- ``schema``: BLADE jsonl record parsing/validation.
- ``data_pipeline``: jsonl logs -> filtered, split (x, y) examples.
- ``features``: lightweight hand-crafted code features (non-RLM baseline).
- ``model``: RegressLM construction, optimizer/schedule, point prediction.
- ``train``: fine-tuning CLI (from-scratch and pretrain-pool+fine-tune).
- ``evaluate``: ranking/selection diagnostics and budget-reduction sim.
- ``report``: assembles ``results/report.md`` from evaluation outputs.
"""

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of [Physical Intelligence's openpi](https://github.com/Physical-Intelligence/openpi) adapted to run the **RoboCasa** benchmark. It trains and serves π0 / π0.5 / π0-FAST vision-language-action models. Code targets Python 3.11 and uses `uv` for dependency management.

The fork's distinguishing feature: RoboCasa training uses **Groot-format datasets** referenced through the external `robocasa` package (`DATASET_SOUP_REGISTRY`, `TASK_SET_REGISTRY`, `get_ds_meta`), not vanilla LeRobot repos. The `robocasa` package must be installed and importable — `config.py` imports it at module load time.

## Commands

```bash
# Install (editable)
pip install -e .
pip install -e packages/openpi-client/

# Lint / format (run before committing)
ruff check .
ruff format .
pre-commit run --all-files     # requires `pre-commit install` once

# Tests (pytest discovers under src/, scripts/, packages/)
uv run pytest                              # or: pytest
uv run pytest src/openpi/models/pi0_test.py            # single file
uv run pytest src/openpi/models/pi0_test.py::test_name # single test
# Tests marked `manual` are skipped by default; run with -m manual
```

### Train → serve → eval workflow

```bash
# 0) Norm stats MUST be computed before training a config (writes to ./assets/<config-name>/<asset_id>)
python scripts/compute_norm_stats.py --config-name=<config-name>

# 1) Train (JAX, FSDP). exp_name names the checkpoint/wandb run.
XLA_PYTHON_CLIENT_MEM_FRACTION=1.0 python scripts/train.py <config-name> --exp-name=<exp-name>
#   resume:    --resume      overwrite existing ckpt dir: --overwrite      disable wandb: --wandb_enabled=False

# 1b) Train (PyTorch, DDP) — same configs/data pipeline, different backend
torchrun --standalone --nnodes=1 --nproc_per_node=<N> scripts/train_pytorch.py <config-name> --exp_name=<exp-name>

# 2) Serve a checkpoint over websocket
python scripts/serve_policy.py --port=8000 policy:checkpoint \
  --policy.config=<config-name> --policy.dir=<checkpoint-path>

# 3) Run RoboCasa rollouts against the server
python examples/robocasa/main.py --args.port 8000 --args.task_set <task-set> \
  --args.split <split> --args.log_dir <checkpoint-path>

# 4) Aggregate eval results
python examples/robocasa/get_eval_stats.py --dir <checkpoint-path>
```

Checkpoints land in `./checkpoints/<config-name>/<exp-name>/<step>`; assets in `./assets/<config-name>/`.

## Architecture

### Config registry is the entry point — `src/openpi/training/config.py`

Everything is driven by a single list, `_CONFIGS`, of frozen `TrainConfig` dataclasses keyed by unique `name`. The CLI (`tyro.extras.overridable_config_cli`) selects one by name and lets you override any field from the command line. `get_config(name)` looks one up in code. When adding an experiment, **append a new `TrainConfig` to `_CONFIGS`** rather than editing scripts.

A `TrainConfig` wires together:
- `model`: a `BaseModelConfig` — almost always `pi0_config.Pi0Config`. Set `pi05=True` for π0.5 (auto-sets `max_token_len=200`, `discrete_state_input=True`); the `model_type` property resolves to `PI0` / `PI05` / `PI0_FAST`.
- `data`: a `DataConfigFactory`. RoboCasa configs use `LeRobotRobocasaDataConfig(data_dirs=DATASET_SOUP_REGISTRY[...])`.
- `weight_loader`: `CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params")` (or `pi05_base`), `PaliGemmaWeightLoader`, or `NoOpWeightLoader` (random init).
- optimizer / LR schedule / `num_train_steps` / `batch_size` / `fsdp_devices` / `ema_decay` etc.

### Data pipeline — four transform stages

A `DataConfigFactory.create()` builds a `DataConfig` with three transform `Group`s applied in order:
1. **`repack_transforms`** — rename raw dataset keys to canonical keys (`observation/image`, `observation/state`, …). Dataset-side only, not at inference.
2. **`data_transforms`** — robot-specific in/out conversion. RoboCasa uses `robocasa_policy.RobocasaInputs` / `RobocasaOutputs` (see `src/openpi/policies/robocasa_policy.py`). Applied in both training and inference. Outputs slice actions to **12 dims**; state is 16-dim padded to model `action_dim`.
3. *(normalization)* — z-score or quantile norm using stats from `norm_stats`.
4. **`model_transforms`** — `ModelTransformFactory`: resize images to 224×224, tokenize the prompt, pad states/actions to `action_dim`. Branches on `model_type`.

`*Inputs` transforms run at **both** train and inference; `*Outputs` run at **inference only**. Keep this symmetry in mind when changing key names — the eval env (`examples/robocasa/main.py`) must feed the same canonical keys the repack transform produces.

### Dataset loading — `src/openpi/training/data_loader.py`

`create_torch_dataset` routes by `DataConfig`:
- `repo_id == "fake"` → `FakeDataset` (the `debug` config).
- `data_dirs` set (RoboCasa) → `GrootOpenpiSingleDataset` (1 dir) or `GrootOpenpiMultiDataset` (mixture, weighted) in `src/openpi/groot_utils/groot_openpi_dataset.py`.
- otherwise → standard LeRobot dataset.

DROID uses an RLDS path (`create_rlds_dataset`, `droid_rlds_dataset.py`). The action sequence length comes from `model.action_horizon`.

Norm stats are loaded in `DataConfigFactory.create_base_config` via `_load_norm_stats(assets_dir, asset_id)`. For Groot datasets there's a fallback that converts stats from the dataset's repo meta. If stats are missing, training raises — run `compute_norm_stats.py` first.

### Models — JAX primary, PyTorch port

- **JAX/Flax NNX** (`src/openpi/models/`) is the reference implementation: `pi0.py`, `pi0_fast.py`, backbone `gemma.py` / `gemma_fast.py` / `siglip.py`, `lora.py`, `tokenizer.py`. `model.py` defines `BaseModel`, `BaseModelConfig`, `Observation`, `ModelType`.
- **PyTorch** (`src/openpi/models_pytorch/`) mirrors it: `pi0_pytorch.py`, `gemma_pytorch.py`. `transformers_replace/` holds patched HuggingFace transformers files (excluded from ruff). Convert JAX→PyTorch weights with `examples/convert_jax_model_to_pytorch.py`.

`scripts/train.py` (JAX) and `scripts/train_pytorch.py` (PyTorch DDP) share the same config + data pipeline.

### Serving & client

`scripts/serve_policy.py` builds a `Policy` (`create_trained_policy`) and exposes it via `WebsocketPolicyServer` (`src/openpi/serving/`). The wire protocol uses msgpack-numpy. The client lives in the separate **`openpi-client`** workspace package (`packages/openpi-client/`) — lightweight, installable on inference machines without the full training stack; `examples/robocasa/main.py` imports it (`openpi_client.websocket_client_policy`, `image_tools`).

## Conventions

- `ruff` with `line-length=120`, single-line isort imports sorted within sections. Print statements are allowed (`T201` ignored).
- Configs and transforms are **frozen dataclasses**; produce modified copies with `dataclasses.replace`, don't mutate.
- JAX training code uses `@at.typecheck` (beartype + jaxtyping) on `train_step` / `init_train_state` — keep array shape/dtype annotations accurate or it raises at runtime.
- `third_party/` (git submodules: aloha, libero) and `transformers_replace/` are excluded from lint/format; don't reformat them.

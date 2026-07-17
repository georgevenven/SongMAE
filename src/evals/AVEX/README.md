# SongMAE in AVEX

This adapter exposes a frozen TinyBird SongMAE checkpoint to AVEX v1.3. AVEX
owns the datasets, cached embeddings, linear probes, retrieval, clustering, and
metrics. This directory only prepares public data and converts audio into
SongMAE embeddings.

The local AVEX checkout and environment are on `disk1`:

```text
/media/george-vengrovski/disk1/avex
/media/george-vengrovski/disk1/avex-venv
```

## Paper protocol

The [AVEX paper](https://arxiv.org/pdf/2508.11845) takes precedence over the
released YAMLs where they differ:

- Frozen final backbone layer, time-mean embedding, and one linear head.
- 900 epochs, AdamW, learning rate `1e-4`, weight decay `0.1`, batch size 32,
  seed 42, and no learning-rate schedule.
- BEANS classification: accuracy, retrieval R-AUC, and known-K clustering NMI.
- BEANS detection and BirdSet: class-macro mAP and retrieval R-AUC.
- Exactly six BEANS classification tasks, five BEANS detection tasks, and seven
  BirdSet tasks. Speech Commands is not part of the paper aggregate.
- Every BEANS detection input is capped at five seconds.

Use these probe parameters in every evaluation config:

```yaml
training_params:
  train_epochs: 900
  lr: 0.0001
  batch_size: 32
  optimizer: adamw
  weight_decay: 0.1
  amp: false
  warmup_epochs: 0
  scheduler_type: none
```

Classification configs use `eval_modes: [probe, retrieval, clustering]`.
Detection and BirdSet configs use `eval_modes: [probe, retrieval]`.

## SongMAE run config

```yaml
model_spec:
  name: songmae
  pretrained: false
  device: cuda
  audio_config:
    sample_rate: 32000
    representation: raw
    normalize: false
    target_length_seconds: 10
    window_selection: center
    extra_config:
      run_dir: /absolute/path/to/TinyBird/runs/songmae_run
      checkpoint: model_step_500000.pth

dataset_config: /media/george-vengrovski/disk1/avex/configs/data_configs/data_base.yml
output_dir: /media/george-vengrovski/disk1/avex_runs/songmae/unused_train
run_name: songmae
label_type: supervised
loss_function: cross_entropy
logging: none
training_params:
  train_epochs: 1
  lr: 0.001
  batch_size: 32
  optimizer: adamw
```

The outer AVEX evaluation config replaces these dummy training parameters.
`checkpoint` is relative to the run's `weights/` directory; omit it to select
the latest checkpoint. Use a 10-second target for BEANS and five seconds for
BirdSet. SongMAE's true context is read from `audio_params.json`.

Only `aggregation: mean` is supported. `last_layer` selects the final encoder
block; explicit layer numbers and `all` remain available for ablations.

## BEANS

AVEX expects an `alp-data` processed BEANS root, not a direct clone of the
BEANS repository. Generate paper-aligned configs with:

```bash
cd /home/george-vengrovski/Documents/projects/TinyBird
PYTHONPATH=/media/george-vengrovski/disk1/avex:$PWD \
  /media/george-vengrovski/disk1/avex-venv/bin/python \
  -m src.evals.AVEX.prepare_beans \
  --root /media/george-vengrovski/disk1/alp-data/beans/v0.1.0/raw \
  --config-dir /media/george-vengrovski/disk1/avex_runs/songmae_beans_all_probe/configs
```

This writes task-level configs plus `beans_classification.yml`,
`beans_detection.yml`, and `beans_all.yml`. Use the first two with their
applicable evaluation modes. Put paper runs in a fresh results directory or
set `overwrite_embeddings: true`; old caches may encode different clip caps.
Set `ALP_DATA_HOME=/media/george-vengrovski/disk1/alp-data` when evaluating so
alp-data resolves the generated local manifests instead of its private GCS root.

## BirdSet

The AVEX repository uses private manifests. The preparer instead pins the
public `mteb/BirdSet` Parquet mirror and materializes local audio and manifests:

```bash
cd /home/george-vengrovski/Documents/projects/TinyBird
PYTHONPATH=/media/george-vengrovski/disk1/avex:$PWD \
  /media/george-vengrovski/disk1/avex-venv/bin/python \
  -m src.evals.AVEX.prepare_birdset \
  --root /media/george-vengrovski/disk1/alp-data/birdset/v0.1.0/raw \
  --cache-dir /media/george-vengrovski/disk1/huggingface/birdset \
  --config-dir /media/george-vengrovski/disk1/avex_runs/songmae_birdset_all_probe/configs
```

It selects `train` and `test_5s` for HSN, UHH, PER, NES, POW, SNE, and NBP,
creates the paper's stratified 80/20 train/validation split, and writes
`birdset_all.yml`. Empty test clips remain all-zero labels: they stay as
negative retrieval candidates but are not valid positive queries.
Completed legacy manifests are repaired before preparation reuse and again
before evaluation, so AVEX never treats their old `None` sentinel as a class.

BirdSet probing uses the paper's environmental-noise and mixup augmentation.
The run config must set the top-level sample rate used by AVEX augmentation:

```yaml
sr: 32000
augmentations:
  - noise:
      noise_dirs:
        - /media/george-vengrovski/disk1/avex_noise/demand_10s
        - /media/george-vengrovski/disk1/avex_noise/idmt
        - /media/george-vengrovski/disk1/avex_noise/tut2016_10s
        - /media/george-vengrovski/disk1/avex_noise/urbansound
        - /media/george-vengrovski/disk1/avex_noise/freesound_10s
        - /media/george-vengrovski/disk1/avex_noise/orcasound_shipnoise_10s
        - /media/george-vengrovski/disk1/avex_noise/deepship_10s
        - /media/george-vengrovski/disk1/avex_noise/shipsear_10s
        - /media/george-vengrovski/disk1/avex_noise/wham_noise
      snr_db_range: [-10, 20]
      augmentation_prob: 0.5
  - mixup:
      alpha: 0.4
      n_mixup: 1
      augmentation_prob: 0.5
```

The nine noise pools follow the paper; the released repository's extra
AudioSet pool is intentionally omitted. The paper does not report mixup alpha
or pair count, so the released-code values are retained. AVEX fails clearly if
any configured noise directory is missing. The eval launcher preserves AVEX's
BirdSet train/validation augmentation while removing its accidental item-level
noise from the test dataset.

Set `AVEX_BIRDSET_ROOT` when evaluating the generated configs:

```bash
AVEX_BIRDSET_ROOT=/media/george-vengrovski/disk1/alp-data/birdset/v0.1.0/raw \
AVEX_EXPERIMENT_DIR=/media/george-vengrovski/disk1/avex_runs/metadata \
PYTHONPATH=/media/george-vengrovski/disk1/avex:$PWD \
  /media/george-vengrovski/disk1/avex-venv/bin/python \
  -m src.evals.AVEX.evaluate --config /absolute/path/to/evaluate.yml
```

## Long clips

Inputs longer than SongMAE's context are split into 50%-overlapping context
windows, including a final window anchored to the clip end. Valid patch tokens
are averaged across windows before AVEX receives one embedding for the original
clip. For a linear head this equals averaging window logits (a soft vote) while
also keeping retrieval and clustering compatible with AVEX's offline cache.

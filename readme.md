# SongMAE

SongMAE is the refactored TinyBird core for training masked autoencoders on
birdsong spectrograms, extracting encoder embeddings, and keeping the run
metadata legible. The old project README is archived as `oldreadme.md`.

> **Note:** Only `src/` and `shell/` reflect the current refactor. Everything
> outside of those two directories is old and in the process of being
> refactored.

The current core is intentionally small. `src/core/` is reserved for the model,
data structures, dataloaders, training, extraction, and small shared utilities.
Dataset converters, embedding runners, external model wrappers, and plotting
helpers live in sibling folders under `src/`.

**Goal: keep `src/core/` under 1500 LoC, excluding comments and blank lines.**
This is a hard ceiling to prevent slop — if a change pushes core over the
limit, simplify or move code into a sibling folder instead of growing core.

## Core Data Files

- `audio_params.json`: stored beside every spectrogram dataset and copied into
  each run folder; defines mel shape, sample rate, hop size, FFT size, mean, and
  std.
- Spectrogram `.npy` files are stored time-major as `(timebins, mels)` for
  faster sequential time access; plotting code transposes them for display.
- `model.json`: stored in each run folder; defines the model architecture and
  fixed input contract.
- `train.json`: stored in each run folder; defines the training recipe used to
  create that run.
- `labels.json` or `*_annotations.json`: stored in `files/annotation jsons/`,
  sometimes beside a corresponding spec folder, and copied into supervised run
  folders.

## Annotation Coverage

A check means the annotation JSON contains that information.

| Species | Detections | Syllable units | Unit onset/offset | Individual ID |
| --- | --- | --- | --- | --- |
| American Robin |  | ✓ |  |  |
| Bengalese Finch |  | ✓ |  |  |
| Canary |  | ✓ |  |  |
| Canary (individual ID) |  |  |  |  |
| Cassin's Vireo |  | ✓ |  |  |
| Chiffchaff |  |  |  |  |
| European Starling |  |  |  |  |
| Great Tit |  |  |  |  |
| Orangutan |  |  |  |  |
| Ovenbird |  |  |  |  |
| Little Owl |  |  |  |  |
| Red-winged Blackbird |  |  |  |  |
| Swamp Sparrow |  |  |  |  |
| Western Capercaillie |  |  |  |  |
| Tree Pipit |  |  |  |  |
| White-crowned Sparrow (manual) |  |  |  |  |
| White-crowned Sparrow (predicted) |  |  |  |  |
| Zebra Finch |  | ✓ |  |  |

## Shell Scripts

- `shell/linear_probe_across_models.sh`: extracts embeddings for syllable-labeled datasets across SongMAE, random SongMAE, AVES, and HuBERT, then runs `src/evals/syllable_classification.py` for each bird/model pair.
- `shell/syllable_classification_train_sweep.sh`: reruns syllable classification from existing embedding folders while sweeping the train-second budget per bird/model.

## Embedding NPZ Contract

Embedding extractors write one concatenated `embeddings.npz` per requested
dataset slice, usually under an output directory for one model and one bird.
This is the sharing and eval format for SongMAE and external models.

Required arrays:

- `encoded_embeddings`: token embeddings, shape `(tokens, dim)`.
- `labels_downsampled`: token labels, shape `(tokens,)`; `-1` is background.
- `recording_stem`: per-token recording stem, shape `(tokens,)`.
- `song_id`: per-token segment id, shape `(tokens,)`.
- `token_start_ms` and `token_end_ms`: per-token time spans in the source
  recording, shape `(tokens,)`.

Optional arrays include `encoded_embeddings_grid` for patch-grid models and
segment-level provenance such as `segment_spec_path`, `segment_wav_path`,
`segment_start_ms`, and `segment_end_ms`.

Downstream tools should load this single file directly. Do not write or depend
on one `.npz` per event.

## Run Folder Layout

Training writes runs under `runs/<run_name>/`:

- `audio_params.json`: copied from the training spectrogram folder.
- `config.json`: legacy combined config for old scripts and checkpoint loading.
- `model.json`: architecture-only config.
- `train.json`: training-only config.
- `logs.txt`: TUI-style training log.
- `weights/`: checkpoint files named `model_step_000000.pth`.
- `imgs/`: optional reconstruction plots.

## `src/core/__init__.py`

- Marks `src.core` as an importable Python package.
- Contains no runtime logic.
- Exists so local scripts can import modules consistently.
- Should stay empty unless the package needs explicit public exports.

## `src/core/audio2spec.py`

- Converts raw audio files into log-mel spectrogram `.npy` files.
- Writes the structural `audio_params.json` metadata for generated spec folders.
- Computes dataset-level mean/std and writes those stats back into
  `audio_params.json`.
- For large datasets, `--storage_dtype int8_affine` writes int8 specs with a
  companion text file per spec that stores per-mel-bin affine scale and offset.
- Int8 affine is preferred over fp8 here: per-mel scaling already handles range,
  so int8 gives 256 uniform levels per mel band instead of fp8's nonuniform
  exponent spacing.
- Can be used as a CLI for full directory conversion with optional stats.

## `src/core/data_structures.py`

- Defines the canonical JSON-backed dataclasses for the refactor.
- `AudioParams` describes spectrogram metadata and normalization stats.
- `ModelConfig` describes architecture fields and derived patch geometry.
- `TrainConfig` describes run/training parameters.
- `Labels` wraps annotation JSON and exposes simple label helpers.

## `src/core/data_loader.py`

- Defines `SpectrogramDataset` for unsupervised SongMAE pretraining.
- Randomly crops or pads spectrograms to `n_timebins`, or returns full files
  when `n_timebins=None`.
- Applies the only supported model normalization: audio-params mean/std.
- Defines `SpectrogramDatasetSupervised`, which can sample full recordings or
  detected-event windows and returns aligned label arrays.

## `src/core/model.py`

- Defines `TinyBird`, the SongMAE masked-autoencoder model.
- Patchifies spectrograms with a strided `Conv2d` and learned 2D positional
  embeddings.
- Supports random masks and Voronoi masks for pretraining.
- Provides encoder inference paths for embedding extraction and layer probing.
- Computes masked-patch reconstruction loss directly on spectrogram patches.

## `src/core/train.py`

- Defines the shared `Trainer` base class for run setup, logging, checkpointing,
  AMP, optimization, and the step loop.
- Defines `UnsupervisedTrainer` for SongMAE reconstruction pretraining.
- Defines a minimal `SupervisedTrainer` scaffold for future classifier-head
  work.
- Writes `config.json`, `model.json`, `train.json`, checkpoints, logs, and
  reconstruction images into each run folder.
- Exposes a CLI for starting or continuing runs.

## `src/core/extract_embedding.py`

- Loads a trained SongMAE checkpoint and extracts encoder embeddings from
  spectrogram recordings.
- Uses `SpectrogramDatasetSupervised` for deterministic event or full-recording
  segment loading.
- Normalizes segments with the run's `audio_params.json`.
- Saves embeddings, downsampled labels, position ids, and optional patch-level
  arrays into `.npz` files.
- Supports optional whitening or PCA-whitening postprocessing for downstream
  UMAP/probe workflows.

## `src/external_models/*.py`

- Wrap AVES, HuBERT, Bird-MAE, and Perch so they use the same recording/event
  windows and JSON labels as SongMAE.
- Save one concatenated `embeddings.npz` per run through
  `src/external_models/data_loader.py`.
- Use `embeddings.tmp.npz` during writes and rename only after extraction
  completes.

## `src/core/utils.py`

- Holds small shared helpers used across dataloading, training, extraction, and
  older scripts.
- Normalizes NumPy arrays and PyTorch tensors using audio-params mean/std.
- Loads checkpoints and returns model state plus model geometry.
- Converts label timestamps between milliseconds and spectrogram timebins.
- Parses annotation events and creates dense per-timebin label arrays.

## `bench_utils/build_wav_manifest.py`

- Builds a JSON manifest mapping spectrogram stems to raw audio files.
- Reads train and validation spec folders to determine required stems.
- Searches a raw audio root for matching `.wav`, `.flac`, `.ogg`, or `.mp3`
  files.
- Fails when stems are missing or ambiguous, so benchmark inputs stay explicit.

## `bench_utils/copy_bird_pool.py`

- Copies or moves all spectrograms for one bird into a separate pool folder.
- Filters recordings from an annotation JSON by optional `bird_id`.
- Copies the source `audio_params.json` into the destination when present.
- Writes `annotations_filtered.json` beside the copied pool.

## `bench_utils/get_ms_per_timebin.py`

- Reads `audio_params.json` from a spectrogram directory.
- Computes milliseconds per spectrogram timebin as `hop_size / sr * 1000`.
- Prints only the numeric value for shell-script use.
- Returns nothing when required metadata is missing.

## `bench_utils/pool_seconds.py`

- Computes total audio duration represented by a spectrogram folder.
- Reads sample rate and hop size from `audio_params.json`.
- Sums the time dimension of all `.npy` spectrogram files.
- Prints total seconds as a single numeric value.

## `bench_utils/sample_by_seconds.py`

- Samples spectrogram files or chunks until a target duration is reached.
- Can filter by annotation JSON and bird id.
- Can create event/unit-centered chunks for supervised benchmarks.
- Can enforce unit coverage, truncate the final sample, and copy or move files.
- Writes matching annotation metadata for sampled outputs.

## `bench_utils/solver_split_by_seconds.py`

- Splits a pool into train/test folders with a mixed-integer solver.
- Balances target train/test duration while preserving unit coverage.
- Handles full files and millisecond-suffixed chunk filenames.
- Writes feasibility details when a feasibility JSON path is provided.
- Copies supporting metadata into the resulting split folders.

## `bench_utils/split_pool_by_duration.py`

- Performs a simpler random duration-based pool split.
- Uses annotation event durations to choose recordings for the held-out set.
- Moves selected spec files into a test folder.
- Copies `audio_params.json` into the test folder when present.

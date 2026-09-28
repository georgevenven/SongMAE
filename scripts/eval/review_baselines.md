# Review baseline preparation

Assets are in `files/review_baselines/`. `manifest.json` records pinned revisions,
download origins, and SHA-256 hashes. `readiness.json` records CPU loading and
synthetic-waveform inference only. No dataset embeddings or evaluation results
were generated during preparation.

- BEATs: `BEATs_iter3_plus_AS2M`, 90,311,792 parameters, no classification head.
  Microsoft source is pinned to `31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd`.
  The official OneDrive download returned HTTP 403, so the checkpoint comes from
  the pinned `Bencr/beats-checkpoints` mirror, with its published SHA-256 verified.
  This verifies mirror-file integrity, not a separate Microsoft-published checksum.
- BirdMAE: official `DBD-research-group/Bird-MAE-Base`, 85,453,056 parameters,
  revision `6cc416d1a7ae2af29b6b866499b3b047a8f01304`.
- Existing environment: `/home/george-vengrovski/anaconda3/envs/mae/bin/python`.
  No environment packages were installed or changed.

Re-download/verify assets without running extraction:

```bash
/home/george-vengrovski/anaconda3/envs/mae/bin/python scripts/prepare_review_baselines.py
```

## Prepared extraction

`src/external_models/review_baselines.py` uses the existing spectrogram-backed
audio dataset, annotation windows, balanced event selection, and embedding writer.
It retains recording/event identifiers and original-time token/segment boundaries.
Eight frequency patches are concatenated into each 6,144-dimensional temporal
embedding; they are not interpreted as eight successive time steps.

| Model condition | Original-time token stride | Maximum original audio per window |
| --- | ---: | ---: |
| BEATs, normal speed | 160 ms | 5 s |
| BirdMAE, normal speed | 160 ms | 5 s |
| BirdMAE, half speed | 80 ms | 2.5 s |
| BirdMAE, quarter speed | 40 ms | 1.25 s |
| BirdMAE, eighth speed | 20 ms | 0.625 s |

Slowdown resamples audio as if played at a lower rate, lowering pitch as well as
speed. Each slowed window fits BirdMAE's official five-second input limit.
Short windows retain the pretrained padding behavior; padded-only output regions
are omitted. Token boundaries follow the 25-ms Kaldi window and 10-ms frame hop,
including their center offset, and are mapped back to original recording time.
Existing span-based scoring can then compare predictions against the original
annotations without treating duplicate upsampled embeddings as independent samples.
The amount of original context decreases with playback speed; it is saved in metadata.

The default is encoder block 11 (zero-based), chosen before any new evaluation.
The extractor also supports `--encoder_layer_idx` or `--all_layers`; all-layer
arrays have shape `[time, layer, feature]` and need explicit layer selection before kNN.
No automatic layer search or model selection is scheduled.

The following command starts extraction and has **not** been run:

```bash
bash shell/extract_review_baselines.sh
```

It inherits the existing `SPEC_ROOT`, `WAV_ROOT`, `ANNOTATION_ROOT`, `PYTHON_BIN`,
`NUM_TIMEBINS` (720000), `FOLDS` (3), and `SEED` (42) defaults. Optional
`DATASET_FILTER`, `BIRD_FILTER`, and `MODEL_FILTER` select subsets. `LAYER` defaults
to 11. Use a distinct `OUT_ROOT` when changing layer or extraction settings;
completed output folders are skipped and existing folders are never overwritten.
Source selection limits are measured in original spectrogram time at every speed.

Outputs follow the existing layout:

```text
results/review_baselines/<dataset>/<bird>/<model>/embeddings/
    encoded_embeddings.npy
    labels_downsampled.npy
    labels_original.npy
    recording_stem.npy
    song_id.npy
    token_start_ms.npy
    token_end_ms.npy
    segment_*.npy
    metadata.json
```

Dataset keys are `canary`, `zf`, and `bf`. Model keys are
`beats_iter3_plus_as2m`, `birdmae_base_speed1`, `birdmae_base_speed0p5`,
`birdmae_base_speed0p25`, and `birdmae_base_speed0p125`.

## Later analysis

The existing kNN CLI accepts prepared embeddings through `--embedding_dir`,
with `--model beats` or `--model birdmae`, `--encoder_layer_idx 11`, and the matching
`--playback_speed`. Its protocol checks now recognize these models and speeds.
No kNN calculations have been run.

Future probe output can remain at `<model>/metrics.json` beside `embeddings/`.
`src/evals/songmae_vs_other_linear_probe_table_aggregator.py` automatically includes
the new model rows when their metrics exist, retaining Canary/Zebra/Bengalese/Mean/
Parsing/Identity columns and TSV/Markdown output. No metrics or table values are fabricated.

Evaluation methodology still needs the separate revision discussed in review:
the current `src/evals/syllable_classification.py` fits PCA over all extracted
tokens before splitting. That code and the selection/evaluation protocol were
not changed as part of model preparation. Resolve fold-local preprocessing and
the held-out evaluation plan before launching comparative probes. Merely reusing
an existing split manifest does not undo prior representation selection.

Paper, figures, masking, and training runs are outside this preparation change.

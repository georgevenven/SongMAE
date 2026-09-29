# SongMAE TMLR revision tracker

Submission 11592. Reviews: [reviews.md](reviews.md). Paper source: [latex/paper.tex](latex/paper.tex) (copied from the submitted build, compiles with `latex/compile.sh`; submitted PDFs in `latex/`). New results go in `results/`, scripts in `scripts/`. Machines: [../servers.md](../servers.md). This file supersedes `../openreview_reviews.md`, which covered only Reviewers 2 and 3.

The project is SongMAE; `TinyBird` is only the old repo folder name.

## Selection protocol (decided)

Model selection must not use the birds it is scored on. Primary protocol: **leave-one-species-out** — select on two species, freeze, evaluate the third; rotate so each species is a test set once (select zf+bf → test canary, etc.). This keeps canary in the test results. Also report the reviewer's **canary-only** variant (select on canary, test zf+bf). The paper must say this is a revised, retrospective analysis: the original selection used all three species.

Everything selected is inside the protocol: masking, patch shape, Voronoi seed %, the Large 32×20 variant, encoder layers (SongMAE and baselines), probe C. Within the test bird, PCA, z-scoring and the probe are fit on training recordings only.

- [ ] Confirm with George: leave-one-species-out as primary, canary-only as secondary.
- [ ] Re-run the replay after the pipeline fixes below.

### Replay on submitted-pipeline results

`scripts/selection_replay.py` → [results/selection_replay_submitted_pipeline.txt](results/selection_replay_submitted_pipeline.txt). kNN purity k=100, averaged over birds within species, then species. Choices that differ from the paper:

| Selected on | Test | Differs from paper |
|---|---|---|
| zf + bf | canary | 20 ms Micro seed → 2.5% |
| bf + canary | zf | 20 ms Micro seed → 2.5% |
| zf + canary | bf | 20 ms Micro seed → 2.5%; BirdAVES layer → L3 |
| canary | zf, bf | 20 ms Micro seed → 2.5%; BirdAVES → L3; HuBERT → L3 |
| each 49 birds (leave-one-bird-out) | the 50th | 20 ms Micro seed → 2.5% (all 50 folds) |

- Masking, patch shape, 5 ms seed %, SongMAE layers (L11 / L10): same as the paper in every rotation, and in 2,000 random bird-level splits.
- **20 ms seed %:** Micro results always favor 2.5%, never the reported 10%. A Large 32×20 at 2.5% was trained (`runs/xcl_large_500k_p32x4_c0025`) but not reported. Choosing between the two Large models picks 10% in every rotation (FER 7.25 vs 7.13; purity 55.2 vs 56.4). **Disclose the 2.5% Large run** and state that the 20 ms seed % was chosen at Large scale.
- **Probe C=0.001:** changed from the default 1.0 after viewing results (commit `3c51c2a`). Leave-one-bird-out picks 0.001 for every model and fold. It helped the baselines more (BirdAVES canary FER 10.07 → 8.23). Disclose it, and re-select on the selection species after the fixes.
- **Baseline layers moving to L3:** on zf/bf purity, L3 is slightly worse for both baselines (BirdAVES 0.621 → 0.615, HuBERT 0.563 → 0.554), so SongMAE's lead should grow by under 1 point. FER and V-measure at L3 have never been run.
- Canary kNN is unaffected by the grouping bug (one event per recording), so the canary-only row is already final for kNN choices.

## Pipeline fixes (do before trusting any new numbers)

- [x] **P1 Probe folds split continuous bouts.** Fixed 2026-09-28: groups are now `recording_stem` in both probes; manifests record `multilabel_stratified_recording`, and old event-grouped manifests are rejected. Watch for `make_folds` asserting on birds where a class appears in fewer than 3 recordings: extraction's `balanced_event_indices` counts events, not recordings. Groups are `stem:song_id`, and `song_id` is the event-segment index (`src/evals/syllable_classification.py:37`, `src/core/extract_embedding.py:74`). bf: 2,832 of 2,965 recordings have more than one event; 90% of neighbouring events touch (0 ms gap), so one bout lands in both train and validation. zf is milder (median gap 629 ms). Canary is unaffected. Contradicts Sec 4.4. Fix: group by `recording_stem` in `syllable_classification.py` and `syllable_classification_capped.py`.
- [x] **P2 kNN excludes only the same event** (fixed 2026-09-28, commit `17435a5`; SongMAE/BirdAVES/HuBERT kNN still to re-run) (`src/embeddings/syllable_knn.py`, `occurrence_neighbors`). Neighbours can come from the adjacent chunk of the same bout. Contradicts A.4. Fix: exclude the same recording.
- [x] **P3 Probe PCA fit on all tokens, validation included** Fixed 2026-09-28: `fold_features` fits PCA on training recordings per fold; `--pca_cache` removed everywhere. Label-budget probe: PCA on all training-recording audio, z-scoring and the probe on labeled tokens only (decided; stated in Sec 4.4). (`pca_fit_scope: all_extracted_tokens`; one shared cache across folds). Fit per fold on training recordings; drop the shared `pca_cache`. For label-limited probes, decide and state whether PCA uses all training audio or only the labeled subset.
- [ ] **P4 Table 4 k-means used stale layers.** Embedding metadata: SongMAE 32×5 L10, 32×20 L9, BirdAVES L6, HuBERT L0; the paper states L11 / L10 / L7 / L0. `shell/syllable_umap_50birds_4models.sh` now says 11, but `--reuse` kept the old embeddings. Micro and Base at L5 are correct. Re-extract at the protocol-selected layers.
- [ ] **P5 Methods text mismatches.** kNN z-scores on reference tokens, not "all embeddings" (A.4). Data per bird differs by analysis: kNN uses the first 200k timebins (1,000 s) in file order; the probe uses up to 720k timebins (60 min) of class-balanced events; k-means uses 250k timebins (~21 min). State this.
- [ ] **P6 5 ms justification uses the test species** (Table S1). Report per species. Canary alone: 0.27% of gaps < 5 ms vs 24.2% at 20 ms.
- [ ] **P7 zf minutes per bird.** Data: 1.3–5.5 min, about 2 min mean, so the tex's "∼2 ± 1" is right and the bioRxiv PDF's "∼5 ± 2" is wrong. Recompute all Sec 4.1 counts from the annotation JSONs.

- [x] **P8 Probe scored only classes present in the model's token labels.** A class too short to win any output bin vanished from the tokens, and its frames were left out of Macro FER. Majority-bin check over all 50 birds: at 20 ms, canary llb11 class 20 vanishes (affects the submitted Table 3 for SongMAE 32×20, BirdAVES and HuBERT, slightly in their favour); at 160 ms, 8 birds. Fixed 2026-09-28: folds are built from each recording's ground-truth classes (identical for every model on a bird) and FER is scored over all ground-truth classes; models train on whatever classes their tokens contain. Tested with a simulated vanished class: it now scores 100% FER instead of being dropped.

Verified, no fix needed: every Table 2 run is SongMAE-Micro (128-d, 6 layers) trained for 100,000 steps (`runs/*micro_100k*/train.json`).

## Compute plan

| Machine | Jobs |
|---|---|
| Work desktop | Micro pretraining: 2 extra seeds per Table 2 row (R1); contiguous-mask baseline (R2); masking-ratio sweep (R2) |
| Twins | Base/Large pretraining, if any is needed |
| Mac Studio | Embedding extraction, kNN (Micro sweep and layer sweeps incl. Large 2.5%), probes, capped probes, k-means, oracle, BEATs / Bird-MAE (normal, ½, ¼, ⅛ speed), syllable counts |

Do extraction, kNN, probes and k-means in **one pass** after P1–P4: every model × layer candidate × bird, with kNN sampling seeds and fold-level outputs saved, so the replay, R1's spread, and the final tables all come from the same run.

## Reviewer 1

- [ ] **Pretraining seeds:** 2 more seeds for random vs Voronoi (32×5, 10%), the five patch shapes, and the seed % settings (32×5 at 2.5/5/10%; 32×20 at 2.5/5/10%). Report mean ± spread in Table 2. Original runs were unseeded (count them as seed 0); new runs use `--seed 1/2`, which seeds torch per rank and the DDP data order.
  - [ ] 32×5 (random, 2.5%, 5%, 10%) × seeds 1–2: **running on Twins** since 2026-09-28 (`scripts/train_micro_seeds.sh`, user unit `songmae-micro-seeds-20260928b`, wandb project `SongMAE-TMLR-revisions` (group `micro_seeds`), log `~/Documents/SongMAE-reviews/logs/micro_seeds.log`). About 2h45m per run → about 22 h for 8. Config matches the original runs except run name, seed and data path.
  - [ ] 128×5, 16×5, 32×20 (2.5/5/10%), 4×20 × seeds 1–2.
- [ ] **Evaluation spread:** folds, kNN sampling seed, and birds for the main results.
- [ ] **Oracle FER:** majority ground-truth label per output bin on 5 ms and 20 ms grids, same 1 ms expansion, with the parsing/identity split. Appendix, referenced from Sec 6.
  - [x] Computed 2026-09-28 (`src/evals/syllable_oracle.py`, `scripts/oracle_all.sh`, `scripts/oracle_table.py` → [results/oracle_fer.md](results/oracle_fer.md)) at 5/20/40/80/160 ms, same data selection and scoring as the probes. Found that the **majority oracle is not a lower bound** on Macro FER: the 20 ms models in Table 3 beat it (7.1–7.3 vs 9.94), because it never predicts classes shorter than half a bin. Added a **macro-optimal** oracle (label maximizing class frames in bin / class frames), which is the true bound: 0.65 / 2.72 / 5.15 / 9.03 / 16.31 % at 5/20/40/80/160 ms. With the submitted Table 3, about 80% of the 5-vs-20 ms parsing gap (2.39 pts) is quantization (bound gap 1.94 pts). Recompute that ratio after the probe re-run.
  - [x] Appendix written 2026-09-28: new **A.2 Oracle Macro FER at output resolution** with Supplemental Figure 1 (lower bound only) and the sentence on why majority isn't a bound. Later appendices renumbered: Voronoi masks A.3, pretraining exclusion (taxonomy) A.4, kNN protocol A.5, BEANS A.6; supplemental figures 2–4. Still to do: reference it from Sec 6, and add the observed-vs-floor comparison after the probe re-run (`% REVISION` comment in A.2). (also answers R3 on coarse models: ≥16% FER floor at 160 ms).
- [ ] **Micro and Base linear-probe FER** at 5 ms and 20 ms (Table 3).
- [ ] **Bird-MAE syllable-level:** Macro FER (parsing/identity) and kNN purity. Covered by R3's coarse-model item.
- [ ] **Speech/music:** 1–2 sentences in the Discussion on Voronoi masking as a remedy for the interpolation shortcut of fine patches.

## Reviewer 2 (64vF)

- [ ] **Contiguous masking baseline:** **Running on the work desktop** since 2026-09-28: time and frequency masking, Micro 32×5, 75%, 100k steps, seeds 0–2 (6 runs). 1 GPU at batch 128 (the Table 2 runs used 2×64 DDP; same global batch). About 5.8 h per run, all done around Wed Sep 30 02:00. User unit `songmae-masking-baselines-20260928`, log `logs/masking_baselines.log`, wandb group `masking_baselines`. Definitions (`src/core/model.py`): **time** masks full-frequency time spans placed by 1D Voronoi over columns, with seed probability C×H per column (C = 5%), so it has the same expected seeds per clip as 2D Voronoi; spans average 14 columns (70 ms) vs 15 for Voronoi's full-frequency spans. **frequency** masks 3 of the 4 frequency bands over the whole clip, leaving a random band visible. Compare with Voronoi 5% and random, all 3 seeds each.
  - Original request: one representative (block, time-only or frequency-only), matched to Micro 32×5 in size, patch, ratio, steps and eval. Compare with random and Voronoi. Narrow the Voronoi claim if contiguous does as well.
- [ ] **Masking-ratio sensitivity:** e.g. 50 / 75 / 90% at fixed patch and seed %, Micro. Report parsing and/or reconstruction.
- [ ] **Sec 3.1:** present in processing order (spectrogram → patch embedding + positions → masking + conv → encoder → decoder / reconstruction), with motivations separated and the changes from standard MAE called out.
- [ ] **Related work:** compare data regimes, temporal granularity and downstream tasks with Ghaffari et al. 2026, Liu et al. 2026 (CVPRW) and Zhang et al. 2025. Verify the bib entries.
- [ ] **Human speech / privacy in XCL:** state what is known and what screening was actually applied (upstream BirdSet/Xeno-Canto and ours). Don't claim screening without evidence.

## Reviewer 3 (j4Qt)

Major:
- [ ] **Stronger frame-level SSL baselines:** BEATs from Miron et al., on the same protocol. BEATs iter3+ AS2M and Bird-MAE-Base are downloaded and verified (`files/review_baselines/manifest.json`, `scripts/eval/review_baselines.md`). Only synthetic CPU inference has been run so far. Drop or substantiate the claim that BirdAVES is the "strongest available baseline".
- [ ] **Coarse models, upsampled:** Bird-MAE at normal speed on the same 1 ms timeline.
  - [ ] **Layer sweep running on the Mac Studio** (started 2026-09-28): kNN at all 12 layers for BEATs 1× and Bird-MAE 1×, ½×, ¼×, ⅛×, on all 50 birds, with same-recording neighbours excluded (P2). Script `scripts/knn_layer_sweep_review_baselines.sh`, log `~/Documents/SongMAE-reviews/logs/knn_layer_sweep_review_baselines.log`, results `results/knn/review_baselines_all_layers/`. Layers are then chosen on the selection species under the protocol, as for BirdAVES and HuBERT; probes use only the chosen layer. Decided: sweep layers (not fixed L11); Bird-MAE only, no original Audio-MAE. First sweep finished 2026-09-28 16:30 (3,000/3,000). **Added 2026-09-28:** BEATs at ½× and ¼× (80, 40 ms steps) and Bird-MAE at 1/16× (10 ms steps; its 5 s window then holds only 0.31 s of original audio, and song at 2–8 kHz moves to 125–500 Hz). Sweep running (log `logs/knn_layer_sweep_review_baselines_extra_speeds.log`). Bird-MAE at **1/32×** (5 ms steps, matching SongMAE 32×5; 0.16 s of original audio per window, song at 62–250 Hz) queued to start when that run finishes (log `logs/knn_layer_sweep_review_baselines_speed1_32.log`).
- [ ] **Half-speed audio:** *Slowdown method fixed 2026-09-28 (commit `218e50e`).* The old code resampled to the model rate first (16 kHz BEATs, 32 kHz Bird-MAE), then slowed, so content above the model's Nyquist (8 kHz for BEATs) was discarded before the slowdown could bring it into range. Synthetic check: a 10 kHz tone at 44.1 kHz vanished under BEATs ½×; now it appears at 5 kHz. Now: resample each recording to `model_sr / speed` and feed the samples as `model_sr` (tape-style: duration ×1/speed, pitch ×speed). Source rates: canary and zf 44.1 kHz, bf 32 kHz. Normal speed unaffected (verified). All slowed conditions deleted and re-running on the Mac (log `logs/knn_layer_sweep_review_baselines_slowed_fixed.log`); normal-speed results kept. **Paper wording:** "we resample each recording to f/s and present it to the encoder at its native rate f, which slows playback by 1/s and lowers every frequency by s, bringing content up to f/(2s) within the encoder's band."
  - BirdAVES slowed too (added 2026-09-28): ½× (10 ms frames) and ¼× (5 ms), same tape-style method, 5 s of model input per window as for the other baselines. Weights are the same file the paper used (`files/birdaves-biox-base.torchaudio.pt`, SHA-256 `3427869c…` on the desktop and the Mac). 1× output is byte-identical to the old code. Queued on the Mac after the BEATs/Bird-MAE run, together with a 1× BirdAVES re-sweep under the recording-level kNN fix (log `logs/knn_layer_sweep_birdaves.log`).
  - Original note: Bird-MAE at ½ speed (also ¼ and ⅛, fixed in advance, not chosen on test data). Note the pitch shift and shorter original-time context per window.
- [ ] **Model-selection leakage:** see "Selection protocol" above. Revise Secs 5–7: the tables mark which species each result was selected on.
- [ ] **Concrete parsing measure:** predicted vs true syllable counts, plus event-level precision/recall/F1 with a boundary tolerance fixed in advance. Keep parsing FER.

Minor:
- [ ] p2 "smaller models": name the sizes. Also soften the claim that Micro 5 ms beats Large 20 ms on clustering: 0.503 vs 0.501 overall; on zf, Micro loses.
- [ ] p3: attribute findings to the authors (Niizumi et al., Baade et al., Wu et al., Huang et al.), not to the models.
- [ ] Positional embeddings: learned, factorized frequency and time embeddings added to patch tokens (`src/core/model.py:194`). The decoder reuses them through `encoder_to_decoder`.
- [ ] p5 "query-key norm" → "query-key normalization".
- [ ] 1.88% statistic: name the species and the averaging (see P6).
- [ ] 100k vs 500k: state that every Table 2 entry is Micro at 100k steps (verified), and that the 500k models are separate.
- [ ] Table 1: add encoder-only parameter counts.
- [ ] Sessions/days: canary recordings span about 10 days per bird (llb3 Apr 23–May 3, llb11 May 4–14, llb16 May 3–11, 2018, from filenames). zf is mostly one day per bird (serial date in filename). bf filenames have no dates, so check Koumura & Okanoya 2016.
- [ ] Syllable labeling: who labeled, and whether classes depend on acoustics or sequence context, per dataset.
- [x] PCA wording/scope (Sec 4.4, Sec 7.2 text updated 2026-09-28): "project onto the top 128 principal components", fit per bird × model on training recordings of each fold (P3). k-means has no folds, so state its scope separately.
- [ ] Define the Voronoi seed %: each patch is a seed with probability C (Bernoulli, `voronoi_mask`), distinct from the 75% masking ratio.
- [ ] "Ablation" → "hyperparameter sweep" / "model selection" throughout.
- [ ] Sec 5.3: which species select layers (per protocol), and the averaging (birds within species, then species equally).
- [ ] BirdNET: justify it as a reference, acknowledge possible BEANS overlap, don't call it the best supervised model; consider a stronger supervised comparator from Miron et al.
- [ ] p11: replace "a sort of trajectory of neural activations" with "the time-ordered sequence of embeddings".
- [ ] One species order everywhere (canary → zebra finch → Bengalese finch).
- [ ] A.4 (was A.3): eBird/Clements taxonomy version, from the BirdSet label provenance.

## Other items from the audit

- [ ] **Unreported sweep candidates** on Twins (`~/Documents/SongMAE/runs`), all at 100k steps: Micro `p32x1_c020`, `p16x1_{c0025,c005,c010,c020,random}`, `p128x1_c005`, `p32x4_c020`, `p32x4_qknorm_gelu`. Also Base 100k (`p16x1`, `p16x4_c010`, `p32x1_c005`, `p32x1_c010`, `p32x4_c010`, `p4x4_c010`), Large 100k `p32x1_c005`, Tiny 100k (`p16x1_c010`, `p32x1_c010`), and Large 500k `p32x4_c0025`. Work out which were evaluated, then include or disclose them in the sweep.
- [ ] Sec 6 opening "Knowing the optimal SongMAE configuration and size": size was fixed in advance, not selected. Reword it.
- [ ] State that the reported Large models use the final 500k checkpoint, with no checkpoint selection.
- [ ] Voronoi mask is one mask per batch, shared across samples: check the text matches `voronoi_mask` usage in `train.py`.

## Edit log

- 2026-09-29 — Mac Studio can't hold the SongMAE-Large all-layer extraction: a user `llama-server` (19 GB, plus most of the ~38 GB wired memory) leaves about 20 GB; the extractor plateaued at ~32 GB with 20+ GB swap (flushing the memmap every 50 segments or every segment and MPS cache limits didn't help). Moved the SongMAE + HuBERT kNN sweep to the work desktop (GPU shared with the frequency-masking training), unit `songmae-knn-sweep-songmae-hubert`. The probe pass is queued on the desktop to start after it (unit `songmae-probes-after-sweep`: `select_layers.py`, then `probe_all.sh`, log `logs/probes.log`). Added P8 (score all ground-truth classes) before any probe ran.
- 2026-09-28 — Mac ran out of RAM on Bird-MAE 1/16 all-layer extraction (whole-bird arrays held in memory, ~30 GB at 1/16 and ~60 GB at 1/32, doubled while saving). Fixed: features stream to one append-only file and are memory-mapped back (`review_baselines.py`), and large arrays are concatenated on disk (`data_loader.concatenate_on_disk`). A first fix using per-chunk memmaps hit macOS's 256 open-file limit. Output verified identical to the reference; 1/32 stress test (about 700 chunks) passed. Sweep relaunched 16:50 for BEATs ½/¼ and Bird-MAE 1/16, 1/32. BEATs is 160 ms per step at 1× (16×16 patches on a 10 ms-hop fbank; checked in source and in the extracted token widths).
- 2026-09-28 — Paper: added appendix A.2 (oracle lower bound + Supplemental Figure 1); renumbered A.3–A.6 and Supplemental Figures 2–4 and their in-text references. No other text changed.
- 2026-09-28 — Oracle FER computed for all 50 birds (majority and macro-optimal) at 5–160 ms.
- 2026-09-28 — Added `time` and `frequency` mask types; launched the 6 masking-baseline runs on the work desktop.
- 2026-09-28 — Probe code fixed (P1 recording-level folds, P3 per-fold PCA) in `syllable_classification.py`, `syllable_classification_capped.py` and `shell/linear_probe_lib.sh`; tested on zf B145. Paper text updated to match: Sec 4.4 (principal-components wording, recording folds, per-fold PCA, C = 10^-3, label-budget PCA scope), Sec 7.2 (k-means PCA scope), A.4 (same-recording exclusion, 1,000 s per bird, z-scoring on reference embeddings). Every affected table and figure is marked `% REVISION: numbers pending re-run` in `paper.tex`.
- 2026-09-28 — Mac Studio env ready (Miniforge `mae`, torch 2.6 with MPS). `review_baselines.py` now falls back to MPS. Smoke test on zf B145 (BEATs 1×, Bird-MAE ½×): token times and labels identical to the CUDA desktop; embeddings match to rel. err 2e-5 / 6e-6. Sync script fixed (Mac rsync 2.6.9 takes remote paths literally).
- 2026-09-28 — Mac Studio set up for evals: key auth, `reviews` clone, eval data syncing (specs 22.4 GB, wavs 18 GB, annotations, baseline weights, 17 final checkpoints). Still to do: Python env, and MPS support (extractors and kNN use `cuda if available else cpu`).
- 2026-09-28 — Added `--seed` to training; cloned the `reviews` branch to Twins (`~/Documents/SongMAE-reviews`, with `data` and `files` linked from `../SongMAE`); started the 32×5 Micro seed queue. The work desktop already has XCL and working clean splits in `/media/george-vengrovski/disk1/data`.
- 2026-09-28 — Created this folder: copied the submitted paper source and compiled it unchanged; saved the reviews; wrote this tracker from the code audit; ran the selection replay on submitted-pipeline results.

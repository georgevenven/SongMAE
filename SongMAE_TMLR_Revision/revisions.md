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

- [ ] **P1 Probe folds split continuous bouts.** Groups are `stem:song_id`, and `song_id` is the event-segment index (`src/evals/syllable_classification.py:37`, `src/core/extract_embedding.py:74`). bf: 2,832 of 2,965 recordings have more than one event; 90% of neighbouring events touch (0 ms gap), so one bout lands in both train and validation. zf is milder (median gap 629 ms). Canary is unaffected. Contradicts Sec 4.4. Fix: group by `recording_stem` in `syllable_classification.py` and `syllable_classification_capped.py`.
- [ ] **P2 kNN excludes only the same event** (`src/embeddings/syllable_knn.py`, `occurrence_neighbors`). Neighbours can come from the adjacent chunk of the same bout. Contradicts A.4. Fix: exclude the same recording.
- [ ] **P3 Probe PCA fit on all tokens, validation included** (`pca_fit_scope: all_extracted_tokens`; one shared cache across folds). Fit per fold on training recordings; drop the shared `pca_cache`. For label-limited probes, decide and state whether PCA uses all training audio or only the labeled subset.
- [ ] **P4 Table 4 k-means used stale layers.** Embedding metadata: SongMAE 32×5 L10, 32×20 L9, BirdAVES L6, HuBERT L0; the paper states L11 / L10 / L7 / L0. `shell/syllable_umap_50birds_4models.sh` now says 11, but `--reuse` kept the old embeddings. Micro and Base at L5 are correct. Re-extract at the protocol-selected layers.
- [ ] **P5 Methods text mismatches.** kNN z-scores on reference tokens, not "all embeddings" (A.4). Data per bird differs by analysis: kNN uses the first 200k timebins (1,000 s) in file order; the probe uses up to 720k timebins (60 min) of class-balanced events; k-means uses 250k timebins (~21 min). State this.
- [ ] **P6 5 ms justification uses the test species** (Table S1). Report per species. Canary alone: 0.27% of gaps < 5 ms vs 24.2% at 20 ms.
- [ ] **P7 zf minutes per bird.** Data: 1.3–5.5 min, about 2 min mean, so the tex's "∼2 ± 1" is right and the bioRxiv PDF's "∼5 ± 2" is wrong. Recompute all Sec 4.1 counts from the annotation JSONs.

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
- [ ] **Micro and Base linear-probe FER** at 5 ms and 20 ms (Table 3).
- [ ] **Bird-MAE syllable-level:** Macro FER (parsing/identity) and kNN purity. Covered by R3's coarse-model item.
- [ ] **Speech/music:** 1–2 sentences in the Discussion on Voronoi masking as a remedy for the interpolation shortcut of fine patches.

## Reviewer 2 (64vF)

- [ ] **Contiguous masking baseline:** one representative (block, time-only or frequency-only), matched to Micro 32×5 in size, patch, ratio, steps and eval. Compare with random and Voronoi. Narrow the Voronoi claim if contiguous does as well.
- [ ] **Masking-ratio sensitivity:** e.g. 50 / 75 / 90% at fixed patch and seed %, Micro. Report parsing and/or reconstruction.
- [ ] **Sec 3.1:** present in processing order (spectrogram → patch embedding + positions → masking + conv → encoder → decoder / reconstruction), with motivations separated and the changes from standard MAE called out.
- [ ] **Related work:** compare data regimes, temporal granularity and downstream tasks with Ghaffari et al. 2026, Liu et al. 2026 (CVPRW) and Zhang et al. 2025. Verify the bib entries.
- [ ] **Human speech / privacy in XCL:** state what is known and what screening was actually applied (upstream BirdSet/Xeno-Canto and ours). Don't claim screening without evidence.

## Reviewer 3 (j4Qt)

Major:
- [ ] **Stronger frame-level SSL baselines:** BEATs from Miron et al., on the same protocol. BEATs iter3+ AS2M and Bird-MAE-Base are downloaded and verified (`files/review_baselines/manifest.json`, `scripts/eval/review_baselines.md`). Only synthetic CPU inference has been run so far. Drop or substantiate the claim that BirdAVES is the "strongest available baseline".
- [ ] **Coarse models, upsampled:** Bird-MAE at normal speed on the same 1 ms timeline.
- [ ] **Half-speed audio:** Bird-MAE at ½ speed (also ¼ and ⅛, fixed in advance, not chosen on test data). Note the pitch shift and shorter original-time context per window.
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
- [ ] PCA wording/scope: "project onto the top 128 principal components", fit per bird × model on training recordings of each fold (P3). k-means has no folds, so state its scope separately.
- [ ] Define the Voronoi seed %: each patch is a seed with probability C (Bernoulli, `voronoi_mask`), distinct from the 75% masking ratio.
- [ ] "Ablation" → "hyperparameter sweep" / "model selection" throughout.
- [ ] Sec 5.3: which species select layers (per protocol), and the averaging (birds within species, then species equally).
- [ ] BirdNET: justify it as a reference, acknowledge possible BEANS overlap, don't call it the best supervised model; consider a stronger supervised comparator from Miron et al.
- [ ] p11: replace "a sort of trajectory of neural activations" with "the time-ordered sequence of embeddings".
- [ ] One species order everywhere (canary → zebra finch → Bengalese finch).
- [ ] A.3: eBird/Clements taxonomy version, from the BirdSet label provenance.

## Other items from the audit

- [ ] **Unreported sweep candidates** on Twins (`~/Documents/SongMAE/runs`), all at 100k steps: Micro `p32x1_c020`, `p16x1_{c0025,c005,c010,c020,random}`, `p128x1_c005`, `p32x4_c020`, `p32x4_qknorm_gelu`. Also Base 100k (`p16x1`, `p16x4_c010`, `p32x1_c005`, `p32x1_c010`, `p32x4_c010`, `p4x4_c010`), Large 100k `p32x1_c005`, Tiny 100k (`p16x1_c010`, `p32x1_c010`), and Large 500k `p32x4_c0025`. Work out which were evaluated, then include or disclose them in the sweep.
- [ ] Sec 6 opening "Knowing the optimal SongMAE configuration and size": size was fixed in advance, not selected. Reword it.
- [ ] State that the reported Large models use the final 500k checkpoint, with no checkpoint selection.
- [ ] Voronoi mask is one mask per batch, shared across samples: check the text matches `voronoi_mask` usage in `train.py`.

## Edit log

- 2026-09-28 — Added `--seed` to training; cloned the `reviews` branch to Twins (`~/Documents/SongMAE-reviews`, with `data` and `files` linked from `../SongMAE`); started the 32×5 Micro seed queue. The work desktop already has XCL and working clean splits in `/media/george-vengrovski/disk1/data`.
- 2026-09-28 — Created this folder: copied the submitted paper source and compiled it unchanged; saved the reviews; wrote this tracker from the code audit; ran the selection replay on submitted-pipeline results.

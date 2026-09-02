# Contextual song probe from Qwen boxes

## Idea

A per-token MLP cannot coordinate decisions across time and frequency. Instead, freeze SongMAE and train one small Transformer layer over its complete 2D token grid. Qwen boxes provide dense song/background targets.

```text
spectrogram
    ↓
frozen SongMAE Large 32×4
    ↓  [batch, 4×250, 768]
LayerNorm + Linear(768, 128)
    ↓
one Transformer layer (4 heads, FFN 256)
    ↓
Linear(128, 1) per token
    ↓
4×250 song-probability grid
    ↓
connected components → boxes
```

The contextual head has 265,089 trainable parameters. The 98.6M-parameter SongMAE backbone is frozen.

## Setup

- Backbone: `runs/xcl_large_500k_p32x4_c0025/weights/model_step_499999.pth`
- Teacher: `data/XCL/qwen38_agentic_annotations.jsonl`
- Patch resolution: 32 mel bins × 4 time frames
- Input context: 1,000 frames = 5 seconds
- Token grid per context: 4 × 250
- Loss: unweighted binary cross-entropy
- Optimizer: AdamW, learning rate `1e-3`, weight decay `1e-4`
- Training: 3 epochs, batch size 8
- Split: recording-disjoint 90% train / 10% validation
- Checkpoint selection: lowest validation BCE
- Threshold: selected on 20 validation recordings
- Final test: 21 untouched validation recordings

Qwen boxes are snapped outward to the SongMAE grid. A token is positive if its patch overlaps a box. The head does not receive Qwen boxes at inference.

## Result

Epoch 2 was selected.

| Metric | Untouched test |
|---|---:|
| Precision | 0.6965 |
| Recall | 0.7286 |
| Micro F1 | 0.7122 |
| Recording-macro F1 | 0.6189 |

This improved over the earlier independent 32×1 MLP's untouched micro F1 of about 0.681. The grids differ, so that comparison is directional rather than exact.

Qualitatively, the attention head produces substantially more coherent song regions. Connected components recover the main Qwen events, though narrow low-frequency false positives remain.

## Five-second context

Longer examples are split into independent 5-second forward passes. Their token grids are concatenated at the true boundaries for plotting and box extraction. There is no attention across chunks.

For 32×4, each 5-second chunk produces 250 temporal tokens. A 10-second plot therefore contains 500 temporal tokens. Overlapping windows with center cropping or blending may reduce boundary effects later.

## Box conversion

The calibrated logit threshold produces a binary 2D grid. Eight-connected components become axis-aligned boxes at the native 32×4 resolution. Singleton components are discarded. Reported metrics are token metrics, not box-IoU metrics.

## Capacity regularization ablation

We tried:

- hidden dimension 64;
- no additional positional embeddings;
- dropout 0.2;
- weight decay `1e-2`.

This reduced the optimized head to about 84k parameters but did not clearly help.

| Metric | Original | Regularized |
|---|---:|---:|
| Precision | 0.6965 | 0.6968 |
| Recall | 0.7286 | 0.7316 |
| Micro F1 | 0.7122 | 0.7138 |
| Recording-macro F1 | 0.6189 | 0.6091 |

The regularized model was visually more fragmented. The original 128-dimensional head is preferred.

## Reproduce

```bash
python scripts/train_agentic_attention_head.py
python scripts/plot_agentic_attention_boxes.py
```

Artifacts:

- `runs/song_unit_32x4/attention_head.pt`
- `runs/song_unit_32x4/attention_head_regularized.pt`

## Limitations and next step

Qwen has false negatives, but dense BCE treats every unboxed token as background. Qwen-based evaluation also rewards reproducing Qwen errors. The next useful experiment is to regularize the supervision rather than reduce head capacity: use box interiors as positive anchors, ignore uncertain boundaries, and treat unboxed regions as weak negatives or unlabeled. A small human-labeled holdout is needed to tell whether this improves biological localization rather than teacher agreement.

## Adaptive-review pilot

The current default trains the same frozen Large 32×4 architecture from
`data/XCL/qwen38_adaptive_review_5s_annotations.jsonl`. Target, uncertain, and chorus events are foreground;
non-target biological and noise events are background. Corrected duplicate rows resolve to the latest result.
The initial 87-window teacher set uses a 25% recording-disjoint validation split because 10% contains only two recordings.
The artifact is `runs/song_unit_32x4/adaptive_attention_head.pt`.

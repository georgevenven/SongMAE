# TMLR reviews — SongMAE (submission 11592)

Verbatim requests as received. Reviewer 2 = OpenReview 64vF, Reviewer 3 = OpenReview j4Qt.

## Reviewer 1

**Do 2 more seeds for each micro run that matters**
“In the paper, please report at least three pretraining seeds for each row the claims depend on: Voronoi vs random masking, the patch-shape variants, and the best seed-percentage settings.”

**Report variance of results**
“In addition, please report evaluation-side spread (across folds, the kNN sampling seed, and birds) for the main results, which should require no retraining.”

**Report oracle performance**
“Oracle FER at output resolution. Please report the Macro FER (with parsing/identity split) of an oracle that assigns each output bin the majority ground-truth label, using the same 5 ms and 20 ms output grids and 1 ms expansion as the evaluated models. The gap between the 5 ms and 20 ms oracles bounds how much of the observed parsing gap (1.85 vs 4.24/4.16) is unavoidable quantization error, and how much reflects representation quality. Either outcome supports the paper's thesis, but the reader needs to know which.”

**Add Micro and Base linear-probe FER at 5 ms and 20 ms (would strengthen the work)**
“Micro and Base results in Table 3. Please add linear-probe FER for SongMAE-Micro and -Base at both 5 ms and 20 ms, mirroring Table 4. The paper's argument that temporal resolution can matter more than capacity is currently shown only for clustering; the size-matched pairs would show whether it also holds for FER.”

**Evaluate Bird-MAE on syllable-level tasks (would strengthen the work)**
“Bird-MAE on syllable-level tasks. Bird-MAE is excluded as too coarse, but the paper's evaluation pipeline already expands each model's predictions into 1 ms frames, so it could be run through the same pipeline. Reporting its Macro FER (with parsing/identity split) and kNN purity would turn the motivating claim into a measured result.”

**Discuss relevance to speech and music (optional)**
“(optional) Relevance to speech and music. Consider adding a sentence or two in the discussion on the implications for speech and music MAEs, particularly Voronoi masking as a general remedy for the interpolation shortcut of fine patches.”

Our position: revise all. Oracle added to appendix; variance added.

## Reviewer 2

**Compare Voronoi masking to contiguous masking**
“Consider a controlled comparison between Voronoi-based masking and representative contiguous masking baseline, such as block, time-only, or frequency-only masking, using the existing masking-ablation evaluation setup.”
“The structured masking comparisons in Audio-MAE [1] provide relevant precedents. [1] Huang, Po-Yao, et al. "Masked autoencoders that listen." Advances in neural information processing systems 35 (2022): 28708-28720.”

**Reorganize Section 3.1 around the processing steps**
“Section 3.1 could be streamlined to make the architecture easier to follow. It alternates between the processing pipeline and the rationale for individual components, including prior findings on convolutional inductive bias and sample efficiency. Separating these motivations from the architectural steps, while highlighting the modifications to the standard MAE design, would improve clarity.”

**Test sensitivity to the 75% masking ratio**
“Suggest a small sensitivity study to assess how the pretraining masking ratio affects reconstruction quality and/or syllable parsing.”

**Expand the related work comparison**
“The related work discussion in Section 2 would benefit from a clearer comparison of the data regimes and downstream tasks with recent work on masked pretraining and data-efficient bioacoustic representation learning [2-4].”
“[2] Ghaffari, Houtan, Lukas Rauch, and Paul Devos. "Data-efficient self-supervised algorithms for fine-grained birdsong analysis." Ecological Informatics 96 (2026): 103862.”
“[3] Wuao Liu, Mustafa Chasmai, Subhransu Maji, and Grant Van Horn. "Masked Autoencoders with Limited Data: Does It Work? A Fine-Grained Bioacoustics Case Study". In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 3131–3140, 2026.”
“[4] Zhang, Qin, et al. "Efficient Masked Autoencoder for Birdsong Representation with Applications on Wild Bird Species Classification." Integrative Zoology (2025).”

**Clarify incidental human speech and privacy handling**
“Pretraining uses the BirdSet XCL subset derived from Xeno-Canto (Section 3.3). Could the authors clarify whether these recordings contain incidental human speech or other potentially identifying content, and whether any relevant screening or handling procedures were applied or needed?”

Our position: revise all.

## Reviewer 3

**Evaluate stronger frame-level SSL baselines (major)**
“Page 6: "as the strongest available baseline at the ~100 mil parameter size". This is debatable. In Miron et al. (which you cite), they found just that the original BEATs model did better on most bioacoustics tasks than BirdAVES (Table 3, line 2 of Miron et al.). You should evaluate the best frame-level SSL models from that paper.”

**Evaluate coarse-resolution models by upsampling their outputs (major)**
“Page 6: "We exclude models operating at resolutions too coarse for syllable-level tasks". Since your whole premise is that these models don't work well for syllable level tasks, you should test them rather than just asserting it. Model outputs at a too-low frame rate can just be up-sampled to the frame rate you use for evaluation.”

**Test half-speed audio as a baseline (major)**
“Page 6: A third baseline that you should test is just feeding in the audio to the too-low-frame-rate models at half-speed. This is a commonly used trick when events of interest are too short (or too high frequency) for your pre-trained encoder (see e.g. Robust detection of overlapping bioacoustic sound events, Mahon et al. 2025). This would be the first thing I would try for the tasks you study in your paper, before designing custom pre-training.”

**Address model-selection leakage (major)**
“Section 5: Here, you've used your evaluation datasets to perform model selection. This means that your evaluation datasets are actually validation sets, and your model is fit to them. Therefore, your evaluation sets cannot be used to assess generalization to other datasets, which is what you are claiming to do in Section 6. There are two solutions: you could use the validation data you set aside from BirdSet to perform model selection, or you could reserve one of your three datasets as a validation dataset and use the other two for model evaluation. It would probably make sense to do the latter, and use the canary dataset as your validation set (since you report that this was already presented in a workshop). This wouldn't change the model you select, since the same model is the best for all three of your datasets, but it would mean that you should adjust how you report results in section 6.”

**Add a more concrete measure of syllable parsing (major)**
“Table 3: I think the parsing FER is hard to interpret and might under-sell why your model is good. Parsing error of 4% already seems quite low. But what does this translate to in terms of something more concrete? For example, I think that you could compare the number of syllables predicted by your model, with the number of actual syllables. Possibly, a 4% parsing error translates into something more substantial when you view it this way.”

**Specify what “smaller models” means (minor)**
“page 2: "We find that for some tasks, such as unsupervised clustering, increased temporal resolution lets smaller models beat out larger coarse ones, suggesting that temporal resolution can matter more than capacity for this class of task." Here "smaller models" is vague, what do you mean?”

**Attribute findings to the authors rather than the model (minor)**
“page 3: "General-audio work supports this distinction. MSM-MAE found that shrinking temporal patch width from 16 frames to 8 or 4 improved most downstream tasks, with the largest gains for speech and music (Niizumi et al., 2022)." The construction of this sentence should be fixed (also for sentences later in this paragraph). The model didn't find that shrinking patch width improved downstream performance, the authors did.”

**Specify the positional embeddings (minor)**
“What positional embeddings do you use? there are a lot of options these days.”

**Change “query-key norm” to “query-key normalization” (minor)**
“page 5: "We apply query-key norm" I think should be "normalization"”

**Identify the dataset behind the 1.88% gap statistic (minor)**
“page 5: "We select 5 ms because only 1.88% of inter-syllable gaps are shorter than this resolution". What dataset/species did you measure this on?”

**Clarify or correct the 100k vs 500k training-step comparison (minor)**
“"100,000 steps for the ablation sweeps" -- this introduces number of training steps as a confound in your ablations. For ablations, did you also train your main model for 100k steps or were you comparing to the 500k-step model? If it's the latter, this should be corrected.”

**Add encoder-only parameter counts to Table 1 (minor)**
“Table 1: You should also display the number of parameters for the encoder, since that's the part you're actually interested in for downstream tasks”

**Clarify recording sessions and days per bird (minor)**
“Page 6: Were recordings made in one session per bird, or across multiple sessions/days?”

**Explain how syllables were labeled (minor)**
“Page 6: You cite the original studies, but you should also summarize how syllables were determined. Were they by experts? Is a syllable label determined just by the acoustics of that syllable, or does it depend on its location in the song sequence?”

**Clarify PCA wording, fitting scope, and cross-validation (minor)**
“Page 7: "so we PCA each model to a common dimensionality of d = 128 before probing or clustering". PCA isn't a verb. You could say "we project each embedding to the top 128 principle components of the embedded data". Also, you should clarify if you performed this PCA per-individual, per-dataset, or just per-model. Also, is PCA fit for the training data of each fold of your cross validation?”

**Define the Voronoi seed percentage (minor)**
“Page 7: "Patch shape and the seed percentage interact". You should indicate what seed percentage is.”

**Distinguish hyperparameter sweeps from ablations (minor)**
“Section 5.1 (and throughout): You use the word "ablations" here but I would use "hyperparameter sweep" or "model selection". Normally "ablation" is reserved for experiments done after you've chosen, by removing features that you introduced as part of the method you're proposing. I was actually quite confused by this, since ablations almost always appear after the main results.”

**Clarify Section 5.3 datasets and dataset-size weighting (minor)**
“Section 5.3: What dataset are these results for? If it's for all datasets, do you correct for imbalance in dataset size?”

**Reconsider BirdNET as the only supervised comparison and address possible leakage (minor)**
“Figure 9 (and appendix A5): It's odd to include BirdNET as the only supervised topline, because there are better performing supervised encoders in Miron et al. which you cite. More recent work may have improved on these results. BirdNET is an especially vexing one to include because they do not make public what training data they use, so it's conceivable that there is leakage from the BEANS test set.”

**Clarify “trajectory of neural activations across time” (minor)**
“Page 11: "a sort of trajectory of neural activations across time" It's unclear what this means.”

**Use a consistent dataset order (minor)**
“Throughout, the order of the datasets changes in figures and tables. Please make them consistent.”

**Specify the eBird/Clements taxonomy version (minor)**
“A.3: You should specify which version of the eBird/Clements taxonomy you used, since they change it every year.”

Our position: revise all; use leave-one-out analysis plus linear probes.

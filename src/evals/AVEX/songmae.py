from contextlib import nullcontext
from math import sqrt

import numpy as np
import torch
from avex import register_model_class
from avex.models.base_model import ModelBase

from src.core.audio2spec import compute_spectrogram
from src.core.data_structures import AudioParams
from src.core.utils import load_model_from_checkpoint, normalize_spectrogram
from src.evals.AVEX.windows import pool_embeddings, select_layers, sliding_window_starts, spatial_embeddings


@register_model_class
class SongMAEAVEX(ModelBase):
    name = "songmae"

    def __init__(self, device, audio_config):
        super().__init__(device=device, audio_config=audio_config)
        extra = audio_config.extra_config
        assert extra and extra.get("run_dir"), "audio_config.extra_config.run_dir is required"
        self.backbone, self.config = load_model_from_checkpoint(extra["run_dir"], extra.get("checkpoint"))
        self.audio = AudioParams.from_dir(extra["run_dir"])
        assert audio_config.sample_rate == self.audio.sr
        self.backbone.to(device).eval()
        self.num_layers = self.config["enc_n_layer"]
        self._layer_names = [f"layer_{index}" for index in range(self.num_layers)]
        self.selected_layers = [self.num_layers - 1]
        self.embedding_batch_size = int(extra.get("embedding_batch_size", 8))
        self.spatial_channels = int(extra.get("spatial_channels", 0))
        self.spatial_identity = bool(extra.get("spatial_identity", False))
        self.spatial_time_pool = int(extra.get("spatial_time_pool", 1))
        self.extra = dict(extra)
        assert self.embedding_batch_size > 0
        assert 0 <= self.spatial_channels <= self.config["enc_hidden_d"]
        assert not (self.spatial_channels and self.spatial_identity)
        assert self.spatial_time_pool > 0
        if self.spatial_identity:
            projection = torch.eye(self.config["enc_hidden_d"])
        elif self.spatial_channels:
            generator = torch.Generator().manual_seed(int(extra.get("spatial_seed", 42)))
            projection = torch.randn(
                self.config["enc_hidden_d"],
                self.spatial_channels,
                generator=generator,
            ) / sqrt(self.config["enc_hidden_d"])
        if self.spatial_identity or self.spatial_channels:
            self.register_buffer("spatial_projection", projection.to(device), persistent=False)

    def get_model_layers(self):
        return self._layer_names.copy()

    def register_hooks_for_layers(self, target_layers):
        self.selected_layers = select_layers(target_layers, self.num_layers)
        return [self._layer_names[index] for index in self.selected_layers]

    def get_embedding_dim(self):
        return self.config["enc_hidden_d"] * (self.audio.mels // self.config["patch_height"])

    def _spectrogram_windows(self, wav, padding_mask):
        context_samples = self.config["num_timebins"] * self.audio.hop_size
        specs, valid_timebins, window_counts = [], [], []
        if padding_mask is None:
            lengths = torch.full((wav.shape[0],), wav.shape[1], device=wav.device)
        else:
            assert padding_mask.shape == wav.shape
            lengths = (~padding_mask.bool()).sum(1)

        for owner, length in enumerate(lengths.tolist()):
            assert length > 0
            raw_wav = wav[owner, :length].detach().float().cpu().numpy()
            raw = compute_spectrogram(raw_wav, self.audio.sr, self.audio.fft, self.audio.hop_size, self.audio.mels)
            starts = sliding_window_starts(length, context_samples)
            window_counts.append(len(starts))
            for start_sample in starts:
                start = start_sample // self.audio.hop_size
                valid = min(max(raw.shape[1] - start, 0), self.config["num_timebins"])
                assert valid > 0
                spec = np.zeros((self.audio.mels, self.config["num_timebins"]), dtype=np.float32)
                spec[:, :valid] = raw[:, start : start + valid]
                specs.append(normalize_spectrogram(spec, self.audio.mean, self.audio.std))
                valid_timebins.append(valid)

        specs = torch.from_numpy(np.stack(specs)).unsqueeze(1).to(self.device)
        valid_timebins = torch.tensor(valid_timebins, device=self.device)
        return specs, valid_timebins, window_counts

    def _encode(self, wav, padding_mask, freeze_backbone, aggregation):
        if freeze_backbone:
            self.backbone.eval()
        specs, valid_timebins, window_counts = self._spectrogram_windows(wav, padding_mask)
        height = self.audio.mels // self.config["patch_height"]
        width = self.config["num_timebins"] // self.config["patch_width"]
        chunks = []
        context = torch.no_grad() if freeze_backbone else nullcontext()
        with context:
            for start in range(0, specs.shape[0], self.embedding_batch_size):
                end = start + self.embedding_batch_size
                valid = valid_timebins[start:end]
                if len(self.selected_layers) == 1:
                    encoded, _ = self.backbone.forward_encoder_inference(
                        specs[start:end],
                        encoder_layer_idx=self.selected_layers[0],
                        valid_timebins=valid,
                    )
                    encoded = encoded[:, None]
                else:
                    encoded, _ = self.backbone.forward_encoder_inference(
                        specs[start:end],
                        valid_timebins=valid,
                        return_all_layers=True,
                    )
                    encoded = encoded[:, self.selected_layers]
                chunks.append(encoded)

        encoded = torch.cat(chunks)
        windows, layers, _, hidden = encoded.shape
        encoded = encoded.reshape(windows, layers, height, width, hidden)
        valid_columns = torch.div(
            valid_timebins + self.config["patch_width"] - 1,
            self.config["patch_width"],
            rounding_mode="floor",
        ).clamp(max=width)

        if aggregation == "none":
            assert layers == 1 and all(count == 1 for count in window_counts)
            assert hasattr(self, "spatial_projection") and width % self.spatial_time_pool == 0
            return spatial_embeddings(
                encoded[:, 0],
                valid_columns,
                self.spatial_projection.to(encoded.dtype),
                self.spatial_time_pool,
            )

        encoded = encoded.permute(0, 1, 3, 2, 4).flatten(3)
        pooled, start = [], 0
        for count in window_counts:
            end = start + count
            pooled.append(pool_embeddings(encoded[start:end], valid_columns[start:end]))
            start = end
        assert start == specs.shape[0]
        return torch.stack(pooled)

    def extract_embeddings(self, x, *, padding_mask=None, aggregation="mean", freeze_backbone=True):
        assert aggregation in ("mean", "none")
        if isinstance(x, dict):
            padding_mask = x.get("padding_mask")
            x = x["raw_wav"]
        pooled = self._encode(x, padding_mask, freeze_backbone, aggregation)
        if aggregation == "none":
            return pooled
        if len(self.selected_layers) == 1:
            return pooled[:, 0]
        return [pooled[:, index] for index in range(pooled.shape[1])]

    def forward(self, x, padding_mask=None):
        embeddings = self.extract_embeddings(
            x,
            padding_mask=padding_mask,
            freeze_backbone=not self.training,
        )
        return embeddings[-1] if isinstance(embeddings, list) else embeddings

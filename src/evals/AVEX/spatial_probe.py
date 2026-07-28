import torch
from avex.models.probes.base_probes import BaseProbe2D
from avex.models.probes.utils import registry


class SpatialConvProbe(BaseProbe2D):
    def _infer_single_tensor_dim(self, embedding):
        assert embedding.ndim == 4 and embedding.shape[1] > 1
        return embedding.shape[1] - 1

    def build_head(self, channels):
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 64, 3, padding=1),
            torch.nn.GELU(),
        )
        self.classifier = torch.nn.Linear(64, self.num_classes)

    def forward(self, inputs, padding_mask=None):
        embeddings = self._get_embeddings(inputs, padding_mask)
        assert isinstance(embeddings, torch.Tensor) and embeddings.ndim == 4
        mask = embeddings[:, -1:]
        features = self.features(embeddings[:, :-1]) * mask
        pooled = features.sum((2, 3)) / mask.sum((2, 3)).clamp_min(1)
        return self.classifier(pooled)


def install_spatial_probe():
    registry._PROBE_CLASSES["linear"] = SpatialConvProbe

from math import sqrt

import torch
from avex.evaluation import finetune
from avex.models.probes.base_probes import BaseProbe2D
from avex.models.probes.utils import registry

from .lora_probe import AsymmetricFineTuneTrainer, Top1Accuracy


def vertical_tokens(grid):
    assert grid.ndim == 4 and grid.shape[1] > 1
    tokens = grid[:, :-1].permute(0, 3, 2, 1).flatten(2)
    mask = grid[:, -1, 0].bool()
    return tokens, mask


class VerticalAttentionProbe(BaseProbe2D):
    def __init__(
        self,
        base_model,
        layers,
        num_classes,
        device="cuda",
        feature_mode=False,
        input_dim=None,
        aggregation="none",
        target_length=None,
        freeze_backbone=True,
        attention_dim=768,
        num_heads=12,
    ):
        assert base_model is not None and not feature_mode
        assert aggregation == "none" and freeze_backbone
        assert attention_dim % num_heads == 0
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        super().__init__(
            base_model=base_model,
            layers=layers,
            num_classes=num_classes,
            device=device,
            feature_mode=feature_mode,
            input_dim=input_dim,
            aggregation=aggregation,
            target_length=target_length,
            freeze_backbone=freeze_backbone,
        )

    def _infer_single_tensor_dim(self, embeddings):
        assert embeddings.ndim == 4 and embeddings.shape[1] > 1
        return (embeddings.shape[1] - 1) * embeddings.shape[2]

    def build_head(self, input_dim):
        self.projection = torch.nn.Linear(input_dim, self.attention_dim, bias=False)
        self.key = torch.nn.Linear(self.attention_dim, self.attention_dim, bias=False)
        self.value = torch.nn.Linear(self.attention_dim, self.attention_dim, bias=False)
        self.query = torch.nn.Parameter(torch.randn(1, self.num_heads, 1, self.attention_dim // self.num_heads) * 0.02)
        self.norm = torch.nn.LayerNorm(self.attention_dim)
        self.classifier = torch.nn.Linear(self.attention_dim, self.num_classes)

    def forward(self, inputs, padding_mask=None):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device != "cpu"):
            grid = self._get_embeddings(inputs, padding_mask)
            tokens, mask = vertical_tokens(grid)
            tokens = self.projection(tokens)
            batch, time, hidden = tokens.shape
            shape = (batch, time, self.num_heads, hidden // self.num_heads)
            key = self.key(tokens).reshape(shape).transpose(1, 2)
            value = self.value(tokens).reshape(shape).transpose(1, 2)
            query = self.query.expand(batch, -1, -1, -1)
            pooled = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask[:, None, None],
            )
            pooled = pooled.transpose(1, 2).reshape(batch, hidden)
            return self.classifier(self.norm(pooled)).float()


def install_attention_probe():
    original_metric = finetune.get_metric_class

    def get_metric(name, num_classes=None):
        if name == "top1":
            return Top1Accuracy()
        return original_metric(name, num_classes)

    registry._PROBE_CLASSES["linear"] = VerticalAttentionProbe
    finetune.FineTuneTrainer = AsymmetricFineTuneTrainer
    finetune.get_metric_class = get_metric

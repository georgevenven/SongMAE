import torch
from avex.evaluation import finetune
from avex.models.probes.base_probes import BaseProbe2D
from avex.models.probes.utils import registry


class LoRALinear(torch.nn.Module):
    def __init__(self, linear, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.linear = linear
        self.down = torch.nn.Linear(linear.in_features, rank, bias=False)
        self.up = torch.nn.Linear(rank, linear.out_features, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.scale = alpha / rank
        torch.nn.init.kaiming_uniform_(self.down.weight)
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, inputs):
        return self.linear(inputs) + self.up(self.down(self.dropout(inputs))) * self.scale


class LoRAProbe(BaseProbe2D):
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
        freeze_backbone=False,
        dropout_rate=0.05,
    ):
        assert base_model is not None and not feature_mode
        assert aggregation == "none" and not freeze_backbone
        state = torch.load(base_model.extra["teacher_checkpoint"], map_location="cpu", weights_only=True)
        indices = torch.tensor(base_model.extra["teacher_indices"])
        assert len(indices) == num_classes
        base_model.register_buffer("task_head_weight", state["label_head.weight"][indices], persistent=False)
        base_model.register_buffer("task_head_bias", state["label_head.bias"][indices], persistent=False)
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
        self.base_model.requires_grad_(False)
        for layer in self.base_model.backbone.encoder.layers:
            layer.qkv = LoRALinear(layer.qkv, dropout=dropout_rate).to(device)
            layer.out_proj = LoRALinear(layer.out_proj, dropout=dropout_rate).to(device)

    def _infer_single_tensor_dim(self, embeddings):
        hidden = self.base_model.config["enc_hidden_d"]
        assert embeddings.ndim == 4 and embeddings.shape[1] == hidden + 1
        return hidden

    def build_head(self, hidden):
        self.classifier = torch.nn.Linear(hidden, self.num_classes)
        with torch.no_grad():
            self.classifier.weight.copy_(self.base_model.task_head_weight)
            self.classifier.bias.copy_(self.base_model.task_head_bias)

    def forward(self, inputs, padding_mask=None):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device != "cpu"):
            grid = self._get_embeddings(inputs, padding_mask)
            assert isinstance(grid, torch.Tensor) and grid.ndim == 4
            features, mask = grid[:, :-1].mean(2), grid[:, -1:, 0]
            pooled_mask = torch.nn.functional.adaptive_avg_pool1d(mask, 16)
            features = torch.nn.functional.adaptive_avg_pool1d(features * mask, 16)
            features = features / pooled_mask.clamp_min(1 / grid.shape[-1])
            logits = self.classifier(features.transpose(1, 2))
            logits = logits.masked_fill(pooled_mask.transpose(1, 2) == 0, torch.finfo(logits.dtype).min)
            return logits.max(1).values.float()


class AsymmetricLoss(torch.nn.Module):
    def forward(self, logits, targets):
        positive = torch.sigmoid(logits.float())
        negative = (1 - positive + 0.05).clamp(max=1)
        loss = targets * positive.clamp_min(1e-8).log()
        loss += (1 - targets) * negative.clamp_min(1e-8).log()
        weight = (1 - positive * targets - negative * (1 - targets)) ** (targets + 4 * (1 - targets))
        return -(loss * weight).mean()


class AsymmetricFineTuneTrainer(finetune.FineTuneTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.multi_label:
            self.criterion = AsymmetricLoss()
            self.log.log_params({"loss_fn": "asymmetric"})


class Top1Accuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, logits, targets):
        predictions = logits.argmax(1)
        if targets.ndim == 1:
            correct = predictions == targets
        else:
            correct = targets.gather(1, predictions[:, None]).squeeze(1).bool()
        self.correct += int(correct.sum())
        self.total += len(predictions)

    def get_primary_metric(self):
        return self.correct / self.total


def install_lora_probe():
    original_metric = finetune.get_metric_class

    def get_metric(name, num_classes=None):
        if name == "top1":
            return Top1Accuracy()
        return original_metric(name, num_classes)

    registry._PROBE_CLASSES["linear"] = LoRAProbe
    finetune.FineTuneTrainer = AsymmetricFineTuneTrainer
    finetune.get_metric_class = get_metric

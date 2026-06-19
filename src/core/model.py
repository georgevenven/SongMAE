import torch
import torch.nn.functional as F
from torch import nn


class SDPABlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.attn_dropout = dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def attention(self, x, key_padding_mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.nhead, self.head_dim).unbind(2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        attn_mask = None if key_padding_mask is None else ~key_padding_mask[:, None, None, :]
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        return x.transpose(1, 2).reshape(B, T, D)

    def forward_features(self, x, key_padding_mask=None, target_feature_type="end_of_block"):
        x = x + self.dropout1(self.out_proj(self.attention(self.norm1(x), key_padding_mask)))
        ffn_out = self.linear2(self.dropout(F.relu(self.linear1(self.norm2(x)))))
        ffn_out = self.dropout2(ffn_out)
        if target_feature_type == "ffn":
            return x + ffn_out, ffn_out
        assert target_feature_type == "end_of_block"
        return x + ffn_out, x + ffn_out

    def forward(self, x, src_key_padding_mask=None):
        return self.forward_features(x, src_key_padding_mask)[0]


class SDPAStack(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, n_layer):
        super().__init__()
        self.layers = nn.ModuleList(
            [SDPABlock(d_model, nhead, dim_feedforward, dropout) for _ in range(n_layer)]
        )

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return x


class SongMAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config["patch_size"]
        self.mask_p = config["mask_p"]
        self.mask_c = config["mask_c"]
        self._grid_cache = {}

        d_enc = config["enc_hidden_d"]
        d_dec = config["dec_hidden_d"]
        patch_dim = self.patch_size[0] * self.patch_size[1]
        max_h = config["mels"] // config["patch_height"]
        max_w = config["num_timebins"] // config["patch_width"]

        self.patch_projection = nn.Linear(patch_dim, d_enc)
        self.patch_mask_token = nn.Parameter(torch.randn(1, d_enc, 1, 1))
        self.patch_conv = nn.Sequential(
            nn.Conv2d(d_enc, d_enc, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_enc, d_enc, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.encoder = SDPAStack(
            d_enc,
            config["enc_n_head"],
            config["enc_dim_ff"],
            config["dropout"],
            config["enc_n_layer"],
        )
        self.encoder_to_decoder = nn.Linear(d_enc, d_dec)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_dec))
        self.decoder = SDPAStack(
            d_dec,
            config["dec_n_head"],
            config["dec_dim_ff"],
            config["dropout"],
            config["dec_n_layer"],
        )
        self.decoder_patch_conv = nn.Sequential(
            nn.Conv2d(d_dec, d_dec, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_dec, d_dec, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.decoder_deconv = nn.ConvTranspose2d(
            d_dec,
            1,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.freq_embed = nn.Parameter(torch.randn(1, d_enc, max_h, 1))
        self.time_embed = nn.Parameter(torch.randn(1, d_enc, 1, max_w))

        self.apply(self._init_weights)
        for layers, n_layer in ((self.encoder.layers, config["enc_n_layer"]), (self.decoder.layers, config["dec_n_layer"])):
            std = 0.02 / (2 * n_layer) ** 0.5
            for layer in layers:
                nn.init.normal_(layer.out_proj.weight, std=std)
                nn.init.normal_(layer.linear2.weight, std=std)
        nn.init.normal_(self.freq_embed, std=0.02 / (2 ** 0.5))
        nn.init.normal_(self.time_embed, std=0.02 / (2 ** 0.5))
        nn.init.normal_(self.patch_mask_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _pos_enc(self, H, W):
        return self.freq_embed[:, :, :H, :] + self.time_embed[:, :, :, :W]

    def valid_token_mask(self, valid_timebins, H, W, device):
        if valid_timebins is None:
            return None
        valid_cols = torch.div(
            valid_timebins.to(device) + self.patch_size[1] - 1,
            self.patch_size[1],
            rounding_mode="floor",
        ).clamp(max=W)
        cols = torch.arange(W, device=device).repeat(H)
        return cols.unsqueeze(0) < valid_cols.unsqueeze(1)

    def voronoi_mask(self, hw, device):
        H, W = hw
        n_masked = round(H * W * self.mask_p)
        seeds = torch.bernoulli(torch.full((H, W), self.mask_c, device=device)).bool()
        seed_coords = torch.nonzero(seeds, as_tuple=False).float()
        if seed_coords.numel() == 0:
            seed_coords = torch.tensor([[H // 2, W // 2]], dtype=torch.float, device=device)

        cache_key = (H, W, device)
        if cache_key not in self._grid_cache:
            y = torch.arange(H, device=device).unsqueeze(1).expand(-1, W)
            x = torch.arange(W, device=device).unsqueeze(0).expand(H, -1)
            self._grid_cache[cache_key] = torch.stack([y, x], dim=-1).float().reshape(-1, 1, 2)

        scale = torch.tensor(self.patch_size, device=device)
        dists = torch.linalg.vector_norm((self._grid_cache[cache_key] - seed_coords.unsqueeze(0)) * scale, dim=2)
        dists = dists.min(dim=1).values
        threshold = torch.kthvalue(dists, min(n_masked, dists.numel())).values
        mask = dists < threshold
        shortfall = n_masked - int(mask.sum().item())
        if shortfall > 0:
            boundary = torch.nonzero(dists == threshold, as_tuple=False).squeeze(1)
            chosen = boundary[torch.randperm(boundary.numel(), device=device)[:shortfall]]
            mask[chosen] = True
        return mask.reshape(H, W)

    def patch_grid(self, x):
        patches = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)(x).transpose(1, 2)
        B = x.size(0)
        H = self.freq_embed.size(2)
        W = self.time_embed.size(3)
        z = self.patch_projection(patches).transpose(1, 2).reshape(B, -1, H, W)
        return z + self._pos_enc(H, W), H, W

    def conv_features(self, x, mask=None):
        z, H, W = self.patch_grid(x)
        if mask is not None:
            z = torch.where(mask.unsqueeze(1), self.patch_mask_token, z)
        return self.patch_conv(z), H, W

    def mask_batch(self, B, H, W, device):
        mask = self.voronoi_mask((H, W), device)
        return mask.unsqueeze(0).expand(B, -1, -1)

    def sparse_restore_indices(self, mask):
        mask = mask[0].flatten()
        keep = torch.nonzero(~mask, as_tuple=False).squeeze(1)
        drop = torch.nonzero(mask, as_tuple=False).squeeze(1)
        restore = torch.cat([keep, drop], dim=0).argsort()
        return keep, restore

    def reconstruct(self, x, valid_timebins=None):
        B = x.size(0)
        z, H, W = self.patch_grid(x)
        mask = self.mask_batch(B, H, W, x.device)
        valid_tokens = self.valid_token_mask(valid_timebins, H, W, x.device)
        loss_mask = mask if valid_tokens is None else mask & valid_tokens.reshape(B, H, W)

        z = torch.where(mask.unsqueeze(1), self.patch_mask_token, z)
        z = self.patch_conv(z).flatten(2).transpose(1, 2)
        keep, restore = self.sparse_restore_indices(mask)
        keep_padding = None if valid_tokens is None else ~valid_tokens[:, keep]
        h = self.encoder(torch.index_select(z, 1, keep), src_key_padding_mask=keep_padding)

        d = self.encoder_to_decoder(h)
        D = d.size(2)
        mask_tokens = self.mask_token.expand(B, H * W - d.size(1), D)
        d = torch.cat([d, mask_tokens], dim=1)
        d = torch.index_select(d, 1, restore)
        d = d + self.encoder_to_decoder(self._pos_enc(H, W).flatten(2).transpose(1, 2))

        key_padding = None if valid_tokens is None else ~valid_tokens
        d = self.decoder(d, src_key_padding_mask=key_padding)
        d = d.transpose(1, 2).reshape(B, D, H, W)
        pred = self.decoder_deconv(self.decoder_patch_conv(d))
        pred = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)(pred).transpose(1, 2)
        return pred, loss_mask.flatten(1)

    def forward(self, x, valid_timebins=None):
        pred, mask = self.reconstruct(x, valid_timebins)
        return self.loss_mse(x, pred, mask)

    def _forward_encoder_layer(self, layer, x, target_feature_type, key_padding_mask):
        return layer.forward_features(x, key_padding_mask, target_feature_type)

    def forward_encoder_inference(
        self,
        x,
        encoder_layer_idx=None,
        average_top_k=None,
        target_feature_type="end_of_block",
        valid_timebins=None,
    ):
        z, H, W = self.conv_features(x)
        z = z.flatten(2).transpose(1, 2)
        valid_tokens = self.valid_token_mask(valid_timebins, H, W, z.device)
        key_padding = None if valid_tokens is None else ~valid_tokens

        layer_features = []
        out = z
        for layer in self.encoder.layers:
            out, feature = self._forward_encoder_layer(layer, out, target_feature_type, key_padding)
            layer_features.append(feature)

        if average_top_k is not None:
            top_k = int(average_top_k)
            assert 0 < top_k <= len(layer_features)
            features = [F.layer_norm(feature, (feature.shape[-1],)) for feature in layer_features[-top_k:]]
            return torch.stack(features, dim=0).mean(dim=0), z

        if encoder_layer_idx is None:
            return out, z

        idx = int(encoder_layer_idx)
        if idx < 0:
            idx += len(layer_features)
        assert 0 <= idx < len(layer_features)
        return layer_features[idx], z

    def loss_mse(self, x, pred, bool_mask):
        target = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)(x).transpose(1, 2)
        return ((pred - target) ** 2)[bool_mask].mean()

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
        attn_mask = None
        if key_padding_mask is not None and key_padding_mask.any():
            attn_mask = ~key_padding_mask[:, None, None, :]
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
            feature = ffn_out
        elif target_feature_type == "end_of_block":
            feature = x + ffn_out
        else:
            raise ValueError(f"Unsupported target_feature_type: {target_feature_type}")
        return x + ffn_out, feature

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
        self.mask_type = config.get("mask_type", "voronoi")
        self.normalize_patches = config.get("normalize_patches", True)
        self._grid_cache = {}

        self.patch_projection = nn.Conv2d(
            in_channels = 1,
            out_channels = config["enc_hidden_d"],
            kernel_size = self.patch_size,
            stride = self.patch_size 
        )

        self.encoder = SDPAStack(
            config["enc_hidden_d"],
            config["enc_n_head"],
            config["enc_dim_ff"],
            config["dropout"],
            config["enc_n_layer"],
        )
        self.decoder = SDPAStack(
            config["dec_hidden_d"],
            config["dec_n_head"],
            config["dec_dim_ff"],
            config["dropout"],
            config["dec_n_layer"],
        )

        self.encoder_to_decoder = nn.Linear(config["enc_hidden_d"], config["dec_hidden_d"])
        self.decoder_to_pixel = nn.Linear(config["dec_hidden_d"], self.patch_size[0] * self.patch_size[1])

        self.mask_token = nn.Parameter(torch.randn(1, 1, config["dec_hidden_d"]))

        # Calculate max patch grid dimensions for 2D positional encoding
        max_h = config["mels"] // config["patch_height"]
        max_w = config["num_timebins"] // config["patch_width"]
        
        self.pos_enc = nn.Parameter(torch.randn(1, config["enc_hidden_d"], max_h, max_w))

        # GPT-style initialization
        self.apply(self._init_weights)
        # GPT-2 scaled init for residual output projections: std = 0.02 / sqrt(2 * n_layer)
        for layers, n_layer in (
            (self.encoder.layers, config["enc_n_layer"]),
            (self.decoder.layers, config["dec_n_layer"]),
        ):
            std = 0.02 / (2 * n_layer) ** 0.5
            for layer in layers:
                nn.init.normal_(layer.out_proj.weight, std=std)
                nn.init.normal_(layer.linear2.weight, std=std)
        # learned positions / mask token
        nn.init.normal_(self.pos_enc, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        mapped = {}
        for key, value in state_dict.items():
            key = key.replace(".self_attn.in_proj_weight", ".qkv.weight")
            key = key.replace(".self_attn.in_proj_bias", ".qkv.bias")
            key = key.replace(".self_attn.out_proj.", ".out_proj.")
            mapped[key] = value
        return super().load_state_dict(mapped, strict=strict, assign=assign)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            # combined QKV projection is a raw Parameter, not an nn.Linear
            if module.in_proj_weight is not None:
                nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)

    def voronoi_mask(self, hw, p=0.75, c=0.1, device=None):
        """
        bernoulli is imprecise (probably fine)

        made by george and opus 
        """
        H, W = hw
        n_patches = H * W
        n_masked_patches = round(n_patches * p)

        # Step 1: Create seeds
        # create matrix with 0.1 values, bernoulli creates coin flip on each position, each pos has 10 precent chance being seed 
        mask = torch.bernoulli(torch.full((H, W), c, device=device)).bool() 
        
        # Step 2: Distance transform
        # returns coords of seeds, N x 2 (the 2 dimensions being row/col idx)
        seed_coords = torch.nonzero(mask, as_tuple=False).float() # returns coords of True (seeds)
        
        # if zero seeds, unlikely, but if p is low and c is low this is bound to happen in a long train run 
        if seed_coords.shape[0] == 0:
            seed_coords = torch.tensor([[H // 2, W // 2]], dtype=torch.float, device=device) # set a seed cord in the middle 
        
        cache_key = (H, W, device)
        if cache_key not in self._grid_cache:
            y_coords = torch.arange(H, device=device).unsqueeze(1).expand(-1, W)
            x_coords = torch.arange(W, device=device).unsqueeze(0).expand(H, -1)
            self._grid_cache[cache_key] = torch.stack([y_coords, x_coords], dim=-1).float().reshape(-1, 1, 2)
        
        # Scale coordinates by actual patch dimensions for proper Euclidean distance
        patch_height = self.patch_size[0]
        patch_width = self.patch_size[1]
        
        grid_flat = self._grid_cache[cache_key]
        seeds_flat = seed_coords.unsqueeze(0)
        
        scale = torch.tensor((patch_height, patch_width), device=device)
        dists = torch.linalg.vector_norm((grid_flat - seeds_flat) * scale, dim=2)
        
        min_distances, _ = torch.min(dists, dim=1)
        distances = min_distances.reshape(H, W)
        
        # Step 3: Find threshold
        distances_flat = distances.flatten()
        threshold = torch.kthvalue(distances_flat, min(n_masked_patches, distances_flat.numel())).values
        
        # Step 4: Create final mask
        final_mask = distances < threshold
        n_selected = torch.sum(final_mask).item()
        n_needed = n_masked_patches - n_selected
        
        if n_needed > 0:
            boundary_mask = (distances == threshold)
            boundary_indices = torch.nonzero(boundary_mask, as_tuple=False)
            if len(boundary_indices) >= n_needed:
                perm = torch.randperm(len(boundary_indices), device=device)[:n_needed]
                selected_boundary = boundary_indices[perm]
                final_mask[selected_boundary[:, 0], selected_boundary[:, 1]] = True
        
        return final_mask

    def random_mask(self, hw, p=0.75, device=None):
        """
        Randomly mask patches with Bernoulli(p).
        """
        H, W = hw
        return torch.rand((H, W), device=device) < p

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

    def forward_encoder(self, x, inference_mode: bool = False, valid_timebins=None):
        """
        Patchify → add pos enc → mask → Transformer encoder.
        Returns:
          h: (B, keep, D_enc), idx_restore, bool_mask, T
        """

        z = self.patch_projection(x)               # (B, D_enc, H', W')
        B, D, H, W = z.shape

        pos_enc = self.pos_enc[:, :, :H, :W]
        z = z + pos_enc
        z_seq = z.flatten(2).transpose(1, 2)        # (B, T, D_enc)
        T = z_seq.size(1)
        valid_tokens = self.valid_token_mask(valid_timebins, H, W, z.device)

        if inference_mode:
            bool_mask = torch.zeros((B, T), dtype=torch.bool, device=z.device)
            idx_restore = torch.arange(T, device=z.device).unsqueeze(0).expand(B, -1)
            key_padding_mask = None if valid_tokens is None else ~valid_tokens
            h = self.encoder(z_seq, src_key_padding_mask=key_padding_mask)  # (B, T, D_enc)
            return h, idx_restore, bool_mask, T

        mask_type = getattr(self, "mask_type", "voronoi")
        if mask_type == "random":
            mask_grid = self.random_mask((H, W), p=self.mask_p, device=z.device)
        else:
            mask_grid = self.voronoi_mask((H, W), p=self.mask_p, c=self.mask_c, device=z.device)
        bool_mask_flat = mask_grid.flatten()
        bool_mask = bool_mask_flat.unsqueeze(0).expand(B, -1)               # (B, T)
        if valid_tokens is not None:
            bool_mask = bool_mask & valid_tokens

        keep_indices = torch.nonzero(~bool_mask_flat, as_tuple=False).squeeze(1)
        mask_indices = torch.nonzero(bool_mask_flat, as_tuple=False).squeeze(1)

        z_keep = torch.index_select(z_seq, 1, keep_indices)                 # (B, keep, D_enc)
        keep_padding_mask = None if valid_tokens is None else ~valid_tokens[:, keep_indices]

        perm = torch.cat([keep_indices, mask_indices], dim=0)               # kept-first layout
        idx_restore = perm.argsort().unsqueeze(0).expand(B, -1)             # (B, T)

        h = self.encoder(z_keep, src_key_padding_mask=keep_padding_mask)    # (B, keep, D_enc)
        return h, idx_restore, bool_mask, T
    
    def _forward_encoder_layer(self, layer, x, target_feature_type="end_of_block", key_padding_mask=None):
        return layer.forward_features(x, key_padding_mask, target_feature_type)

    def forward_encoder_inference(
        self,
        x,
        encoder_layer_idx=None,
        average_top_k=None,
        target_feature_type="end_of_block",
        valid_timebins=None,
    ):
        z = self.patch_projection(x)               # (B, D_enc, H', W')
        B, D, H, W = z.shape

        pos_enc = self.pos_enc[:, :, :H, :W]
        z = z + pos_enc
        z_seq = z.flatten(2).transpose(1, 2)        # (B, T, D_enc)
        valid_tokens = self.valid_token_mask(valid_timebins, H, W, z.device)
        key_padding_mask = None if valid_tokens is None else ~valid_tokens

        layers = getattr(self.encoder, "layers", None)
        if layers is None:
            raise RuntimeError("SongMAE.encoder does not expose .layers; cannot run encoder inference.")

        layer_features = []
        out = z_seq
        for layer in layers:
            out, feature = self._forward_encoder_layer(
                layer,
                out,
                target_feature_type=target_feature_type,
                key_padding_mask=key_padding_mask,
            )
            layer_features.append(feature)

        if average_top_k is not None:
            top_k = int(average_top_k)
            num_layers = len(layer_features)
            if top_k <= 0 or top_k > num_layers:
                raise ValueError(f"average_top_k out of range: {average_top_k} (num_layers={num_layers})")
            top_features = layer_features[-top_k:]
            normed_features = [F.layer_norm(feature, (feature.shape[-1],)) for feature in top_features]
            h = torch.stack(normed_features, dim=0).mean(dim=0)
        elif encoder_layer_idx is None:
            h = out
        else:
            num_layers = len(layer_features)
            idx = int(encoder_layer_idx)
            if idx < 0:
                idx = num_layers + idx
            if idx < 0 or idx >= num_layers:
                raise ValueError(f"encoder_layer_idx out of range: {encoder_layer_idx} (num_layers={num_layers})")
            h = layer_features[idx]
        return h, z_seq # z seq is encoded patches + pos enc 

    def forward_decoder(self, h, idx_restore, T, valid_timebins=None):
        """
        Project to decoder dim → insert mask tokens → unshuffle → add pos → decode → predict pixels.
        Returns:
          pred: (B, T, P) where P = patch_size[0]*patch_size[1]
        """
        B = h.size(0)
        # project encoder tokens to decoder width
        y = self.encoder_to_decoder(h)                 # (B, keep, D_dec)
        D_dec = self.decoder_to_pixel.in_features
        keep = y.size(1)

        # build full sequence with mask tokens then unshuffle to original order
        mask_tokens = self.mask_token.expand(B, T - keep, D_dec)     # (B, T-keep, D_dec)
        y_full = torch.cat([y, mask_tokens], dim=1)                  # kept-first layout
        y_full = torch.gather(y_full, 1, idx_restore.unsqueeze(-1).expand(B, T, D_dec))

        # Convert 2D pos enc to 1D sequence format for decoder
        # We need to determine H, W from T and the original patch grid dimensions
        H_max = self.pos_enc.size(2)
        W = T // H_max
        # Assume the patches fill the grid in row-major order
        pos_enc_seq = self.pos_enc[:, :, :, :W].flatten(2, 3).transpose(1, 2)  # (1, T, D_enc)
        pos_dec = self.encoder_to_decoder(pos_enc_seq)    # (1, T, D_dec)
        y_full = y_full + pos_dec

        valid_tokens = self.valid_token_mask(valid_timebins, H_max, W, y_full.device)
        key_padding_mask = None if valid_tokens is None else ~valid_tokens
        d = self.decoder(y_full, src_key_padding_mask=key_padding_mask)     # (B, T, D_dec)
        pred = self.decoder_to_pixel(d)                               # (B, T, P)
        return pred

    def forward(self, x, valid_timebins=None):
        h, idx_restore, bool_mask, T = self.forward_encoder(x, valid_timebins=valid_timebins)
        pred = self.forward_decoder(h, idx_restore, T, valid_timebins=valid_timebins)
        return self.loss_mse(x, pred, bool_mask)

    def loss_mse(self, x, pred, bool_mask):
        """
        Compute MSE on masked patches only.
        x:    (B, 1, H, W)
        pred: (B, T, P) from decoder
        """
        unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        target = unfold(x).transpose(1, 2)                            # (B, T, P)
        
        # Optionally normalize target patches
        if self.normalize_patches:
            target_mean = target.mean(dim=-1, keepdim=True)
            target_std = target.std(dim=-1, keepdim=True)
            target = (target - target_mean) / (target_std + 1e-6)
        
        loss = ((pred - target) ** 2)[bool_mask].mean()
        return loss

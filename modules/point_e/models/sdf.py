"""
SDF (Signed Distance Function) model implementation for point cloud processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .transformer import Transformer
from .perceiver import SimplePerceiver
from .util import timestep_embedding

class PointCloudSDFModel(nn.Module):
    """
    A model that predicts signed distance values for points in 3D space.
    """
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        input_channels: int = 3,
        output_channels: int = 1,
        n_ctx: int = 1024,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        time_token_cond: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx
        self.time_token_cond = time_token_cond
        
        self.time_embed = nn.Sequential(
            nn.Linear(width, width * 4, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(width * 4, width, device=device, dtype=dtype),
        )
        
        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.backbone = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx + int(time_token_cond),
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels, device=device, dtype=dtype)
        
        # Initialize output projection to zero
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute signed distance values for input points.
        
        Args:
            x: Input points tensor of shape [batch_size, input_channels, n_points]
            t: Timestep tensor of shape [batch_size]
            
        Returns:
            Tensor of shape [batch_size, output_channels, n_points] containing SDF values
        """
        assert x.shape[-1] == self.n_ctx, f"Expected {self.n_ctx} points, got {x.shape[-1]}"
        
        # Embed timesteps
        t_emb = self.time_embed(timestep_embedding(t, self.backbone.width))
        
        # Project input points to feature space
        h = self.input_proj(x.permute(0, 2, 1))  # [B, N, C]
        
        # Add time embeddings
        if not self.time_token_cond:
            h = h + t_emb[:, None]
        else:
            h = torch.cat([t_emb[:, None], h], dim=1)
        
        # Apply transformer backbone
        h = self.ln_pre(h)
        h = self.backbone(h)
        h = self.ln_post(h)
        
        # Remove time token if used
        if self.time_token_cond:
            h = h[:, 1:]
            
        # Project to output space
        h = self.output_proj(h)
        
        return h.permute(0, 2, 1)  # [B, C, N]

    def compute_sdf(self, points: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        """
        Compute signed distance values for arbitrary points.
        
        Args:
            points: Input points tensor of shape [batch_size, 3, n_points]
            t: Optional timestep tensor of shape [batch_size], defaults to 0
            
        Returns:
            Tensor of shape [batch_size, 1, n_points] containing SDF values
        """
        if t is None:
            t = torch.zeros(points.shape[0], device=points.device)
            
        return self.forward(points, t)

class CrossAttentionPointCloudSDFModel(PointCloudSDFModel):
    """
    Encode point clouds using a transformer, and query points using cross
    attention to the encoded latents.
    """
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 4096,
        width: int = 512,
        encoder_layers: int = 12,
        encoder_heads: int = 8,
        decoder_layers: int = 4,
        decoder_heads: int = 8,
        init_scale: float = 0.25,
    ):
        super().__init__(device=device, dtype=dtype, n_ctx=n_ctx, width=width, layers=encoder_layers, heads=encoder_heads, init_scale=init_scale)
        self._device = device
        self.n_ctx = n_ctx

        self.encoder_input_proj = nn.Linear(3, width, device=device, dtype=dtype)
        self.encoder = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            layers=encoder_layers,
            heads=encoder_heads,
            init_scale=init_scale,
        )
        self.decoder_input_proj = nn.Linear(3, width, device=device, dtype=dtype)
        self.decoder = SimplePerceiver(
            device=device,
            dtype=dtype,
            n_data=n_ctx,
            width=width,
            layers=decoder_layers,
            heads=decoder_heads,
            init_scale=init_scale,
        )
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, 1, device=device, dtype=dtype)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def default_batch_size(self) -> int:
        return self.n_ctx

    def encode_point_clouds(self, point_clouds: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of point clouds to cache part of the SDF calculation.

        Args:
            point_clouds: a batch of [batch x 3 x N] points.
            
        Returns:
            A state representing the encoded point cloud batch.
        """
        h = self.encoder_input_proj(point_clouds.permute(0, 2, 1))
        h = self.encoder(h)
        return dict(latents=h)

    def predict_sdf(
        self, x: torch.Tensor, encoded: Optional[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Predict the SDF at the query points given the encoded point clouds.

        Args:
            x: a [batch x 3 x N'] tensor of query points.
            encoded: the result of calling encode_point_clouds().
            
        Returns:
            A [batch x N'] tensor of SDF predictions.
        """
        data = encoded["latents"]
        x = self.decoder_input_proj(x.permute(0, 2, 1))
        x = self.decoder(x, data)
        x = self.ln_post(x)
        x = self.output_proj(x)
        return x[..., 0]

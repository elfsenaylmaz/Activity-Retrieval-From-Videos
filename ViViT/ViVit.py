import torch
from torch import nn
import torchvision
from torch.utils.data import TensorDataset
import numpy as np
from torch.utils.data.dataloader import DataLoader
from module import *

class VideoVisionTransformer(nn.Module):
    def __init__(
            self,
            img_size=224,
            temporal_size=64,
            patch_size=16,
            in_chans=3,
            n_classes=20,
            embed_dim=768,
            depth=6,
            n_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.2,
            attn_p=0.2,
    ):
        super().__init__()

        self.patch_embed = TubeletEmbedding(
                img_size=img_size,
                temporal_size=temporal_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
                torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)


    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(
                n_samples, -1, -1
        )  # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_token_final = x[:, 0]  # just the CLS token
        x = self.head(cls_token_final)

        return x

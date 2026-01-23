import torch
import torch.nn as nn
import torch.nn.functional as F

# Computed mean:  [0.4980974 0.4980974 0.4980974]
# Computed std:  [0.22967155 0.22967155 0.22967155]

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.n_patches_per_side = (img_size // patch_size) 
        self.n_patches = self.n_patches_per_side**2
        # quindi la proiezione lineare viene fatta con una conv2d, per ottenre 768 valori pixel per patch
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        # costruisco la sequenza dei token che ho
        x = x.flatten(2)  # [B, embed_dim, n_patches]
        # faccio la transposizione per avere la forma corretta che vuole il transformer, cioè [B, n_patches, embed_dim]
        x = x.transpose(1, 2)  # [B, n_patches, embed_dim]
        return x

class SimpleViT(nn.Module):
    def __init__(self, num_classes=15, img_size=224, patch_size=16, in_channels=3, embed_dim=256, num_heads=4, depth=4, mlp_dim=512):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        # il cls token è il classification token, un vettore di embedding che viene aggiunto all'inizio della sequenza dei patch
        # è fondamentalmente un aggregatore al posto di un pooling o di un aggregation function che mi permette di riassumere tutti i vettori di tutte le patches
        # in un solo vettore che poi passo al classificatore finale
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.dropout = nn.Dropout(0.1)

        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, dropout=0.1, batch_first=True)
            for _ in range(depth)
        ])

        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # [B, n_patches, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1+n_patches, embed_dim]
        x = x + self.pos_embed
        x = self.dropout(x)

        for layer in self.transformer_layers:
            x = layer(x)

        cls_out = x[:, 0]  # prendo solo il token cls
        out = self.mlp_head(cls_out)
        return out

if __name__ == "__main__":
    model = SimpleViT(num_classes=15)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)  # [2, 15]

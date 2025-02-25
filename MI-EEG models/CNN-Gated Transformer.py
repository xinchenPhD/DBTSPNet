import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
from torch import Tensor
import torch.nn.functional as F
import math


class PatchEmbedding_time(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),  # temporal conv    in=1,out=k=40  kernel_size=（1,25）,stride=(1,1)
            nn.Conv2d(40, 40, (22, 1), (1, 1)),  # spatial conv in=k=40 ,out=k=40,kernel_size=ch=（22,1),stride=(1,1)
            nn.BatchNorm2d(40),  # relu+BN
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # kernel_size=（1,75）,stride=(1,15)
            nn.Dropout(0.5),  # 防止过拟合
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class PatchEmbedding_channel(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()

        self.shallownet = nn.Sequential(
            # nn.Conv2d(1, 40, (1, 25), (1, 1)),  # temporal conv    in=1,out=k=40  kernel_size=（1,25）,stride=(1,1)
            nn.Conv2d(22, 40, (1, 22), (1, 1)),  # spatial conv in=k=40 ,out=k=40,kernel_size=ch=（22,1),stride=(1,1)
            nn.BatchNorm2d(40),  # relu+BN
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # kernel_size=（1,75）,stride=(1,15)
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GateLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(2 * 40, 1)  # 2 * emb_size, change this based on the actual emb_size  #TODO
        # self.gate = nn.Linear(200, 1)  # 2 * emb_size, change this based on the actual emb_size

    def forward(self, x1, x2):
        b, n, _ = x1.shape
        x1 = rearrange(x1, 'b n e -> b n 1 e')
        x2 = rearrange(x2, 'b n e -> b n 1 e')

        x = torch.cat([x1, x2], dim=-1)  # Concatenate along the last dimension
        gate_weight = torch.sigmoid(self.gate(x))
        # gate_weight = torch.softmax(self.gate(x))
        # Get weights based on the two outputs
        x = gate_weight * x1 + (1 - gate_weight) * x2
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads=10, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__()
        self.attention = MultiHeadAttention(emb_size, num_heads, drop_p)
        self.attention_norm = nn.LayerNorm(emb_size)
        self.attention_dropout = nn.Dropout(drop_p)
        self.feed_forward = FeedForwardBlock(emb_size, forward_expansion, forward_drop_p)

    def forward(self, x, mask=None):
        x_norm = self.attention_norm(x)
        x = self.attention(x_norm, mask)
        x = self.attention_dropout(x)
        x = self.feed_forward(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth, emb_size):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderBlock(emb_size) for _ in range(depth)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2440, 256),  #TODO
            # nn.Linear(6100, 256),  # TODO
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        # x = x.mean(dim=1)  # Global average pooling
        # print("--",x.shape)
        # x = self.fc(x)
        # return x
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class GatedTransformer(nn.Module):
    def __init__(self, emb_size=40, depth=6, n_classes=4):
        super().__init__()
        self.patch_embedding_time = PatchEmbedding_time(emb_size)
        self.patch_embedding_channel = PatchEmbedding_channel(emb_size)
        self.time_step_encoder = TransformerEncoder(depth, emb_size)
        self.channel_encoder = TransformerEncoder(depth, emb_size)
        self.gate_layer = GateLayer()
        self.classification_head = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        # Split the input into time and channel parts
        time_input = x
        channel_input = time_input.transpose(1, 2)  # Swap time and channel axes

        # Ensure the channel input has the correct shape [B, 1, H, W]
        # channel_input = channel_input.unsqueeze(1)  # Add a channel dimension

        # Apply the patch embedding
        time_emb = self.patch_embedding_time(time_input)
        channel_emb = self.patch_embedding_channel(channel_input)

        # Process both encoders
        time_enc = self.time_step_encoder(time_emb)
        channel_enc = self.channel_encoder(channel_emb)

        # Apply the gate layer to combine the two encodings
        combined = self.gate_layer(time_enc, channel_enc)
        # temp = combined
        # Classification head
        output = self.classification_head(combined)

        return output


if __name__ == '__main__':
    x = torch.randn(72, 1, 22, 1000).cuda()
    model = GatedTransformer().cuda()
    y = model(x)
    for i in y:
        print(i.shape)


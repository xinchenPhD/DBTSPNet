import torch
from torch import nn
from torch import Tensor
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
import math


class PatchEmbeddingCNN(nn.Module):
    def __init__(self, f1=16, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.5,
                 number_channel=22, emb_size=40):
        super().__init__()
        f2 = D * f1
        self.cnn_module = nn.Sequential(
            # temporal conv kernel size 64=0.25fs=0.25*256Hz
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False),  # [batch, 22, 1000]
            nn.BatchNorm2d(f1),
            # channel depth-wise conv
            nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1, padding='valid', bias=False),  #
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 1
            nn.AvgPool2d((1, pooling_size1)),
            # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout_rate),
            # spatial conv
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),

            # average pooling 2 to adjust the length of feature into transformer encoder
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),

        )
        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.cnn_module(x)
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


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class ClassificationHead(nn.Sequential):
    def __init__(self, flatten_number, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flatten_number, n_classes)
        )

    def forward(self, x):
        out = self.fc(x)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)

        out = self.layernorm(self.drop(res) + x_input)
        return out


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=4,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                MultiHeadAttention(emb_size, num_heads, drop_p),
            ), emb_size, drop_p),
            ResidualAdd(nn.Sequential(
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
            ), emb_size, drop_p)

        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, heads) for _ in range(depth)])


class BranchEEGNetTransformer(nn.Sequential):
    def __init__(self, heads=4,
                 depth=6,
                 emb_size=40,
                 number_channel=22,
                 f1=16,
                 kernel_size=64,
                 D=2,
                 pooling_size1=8,
                 pooling_size2=8,
                 dropout_rate=0.5,
                 **kwargs):
        super().__init__(
            PatchEmbeddingCNN(f1=f1,
                              kernel_size=kernel_size,
                              D=D,
                              pooling_size1=pooling_size1,
                              pooling_size2=pooling_size2,
                              dropout_rate=dropout_rate,
                              number_channel=number_channel,
                              emb_size=emb_size),
        )


# learnable positional embedding module
class PositioinalEncoding(nn.Module):
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))
        # self.linear = nn.Linear(32, 40)
        # print('111', self.encoding.shape)  #[1, 100, 40]

    def forward(self, x):  # x-> [batch, embedding, length]
        # print('1', x.shape)  # x[72, 15, 32]
        # x_transformed = self.linear(x)
        # print('2', x_transformed.shape)    # x[72, 15, 40]
        # x= self.encoding[:, :x.shape[1], :]   # [1, 15, 40])
        # print('2', x.shape)
        # encoding_transformed = self.linear(self.encoding)
        # print('2', x.shape)
        # print('4',self.encoding[:, :x_transformed.shape[1], :].shape)
        # print('5', self.encoding[:, :x.shape[1], :].shape)
        # x = x_transformed + self.encoding[:, :x.shape[1], :].cuda()
        x = x + self.encoding[:, :x.shape[1], :].cuda()
        # print('3', x.shape)
        return self.dropout(x)


class EEGTransformer(nn.Module):
    def __init__(self, heads=4,
                 emb_size=40,
                 depth=6,
                 # eeg1_f1=16,
                 eeg1_f1=20,
                 eeg1_kernel_size=64,
                 eeg1_D=2,
                 eeg1_pooling_size1=8,
                 eeg1_pooling_size2=8,
                 # eeg1_dropout_rate=0.5,
                 eeg1_dropout_rate=0.3,
                 eeg1_number_channel=22,
                 flatten_eeg1=600,
                 number_class=4,
                 **kwargs):
        super().__init__()
        # self.number_class = 4
        # self.number_channel = numberClassChannel(database_type)
        self.emb_size = emb_size
        self.flatten_eeg1 = flatten_eeg1
        self.flatten = nn.Flatten()
        self.cnn = BranchEEGNetTransformer(heads=4, depth=6, emb_size=40, number_channel=eeg1_number_channel,
                                           f1=eeg1_f1,
                                           kernel_size=eeg1_kernel_size,
                                           D=eeg1_D,
                                           pooling_size1=eeg1_pooling_size1,
                                           pooling_size2=eeg1_pooling_size2,
                                           dropout_rate=eeg1_dropout_rate,
                                           )

        self.position = PositioinalEncoding(emb_size, dropout=0.1)
        self.trans = TransformerEncoder(heads, depth, emb_size)

        self.flatten = nn.Flatten()
        self.classification = ClassificationHead(flatten_eeg1, number_class)   # FLATTEN_EEGNet + FLATTEN_cnn_module

    def forward(self, x):
        cnn = self.cnn(x)
        #  positional embedding
        cnn = cnn * math.sqrt(self.emb_size)
        cnn = self.position(cnn)

        trans = self.trans(cnn)
        # residual connect
        features = cnn + trans

        out = self.classification(self.flatten(features))
        return features, out


if __name__ == '__main__':
    x = torch.randn(72, 1, 22, 1000).cuda()
    model = EEGTransformer().cuda()
    y = model(x)
    for i in y:
        print(i.shape)

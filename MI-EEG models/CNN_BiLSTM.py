import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
from torch import Tensor
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=64):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 64, (1, 25), (1, 1)),  # temporal conv
            nn.Conv2d(64, 64, (22, 1), (1, 1)),  # spatial conv
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # Maybe change pooling to better preserve temporal data
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(64, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),  # Flatten into sequence
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class CNN_Bilstm(nn.Module):
    def __init__(self, emb_size=64, lstm_hidden_size=128, num_layers=2, num_classes=4):
        super(CNN_Bilstm, self).__init__()
        self.embedding = PatchEmbedding(emb_size)

        self.bilstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        lstm_out, _ = self.bilstm(x)
        lstm_out_last = lstm_out[:, -1, :]  # (batch_size, hidden_size * 2)

        out = self.fc(lstm_out_last)
        temp = out
        return temp, out


if __name__ == '__main__':
    x = torch.randn(72, 1, 22, 1000).cuda()
    model = CNN_Bilstm().cuda()
    y = model(x)
    for i in y:
        print(i.shape)



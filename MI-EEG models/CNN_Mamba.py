import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)
# Configuration flags and hyperparameters
USE_MAMBA = 1
DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM = 0
batch_size = 32  # B
last_batch_size = 81  # only for the very last batch of the dataset
current_batch_size = batch_size
different_batch_size = False
h_new = None
temp_buffer = None
torch.cuda.empty_cache()


class EEGNetModel(nn.Module):
    def __init__(self, chans=22, classes=4, time_points=1000, temp_kernel=32,
                 f1=16, f2=32, d=2, pk1=8, pk2=16, dropout_rate=0.5, max_norm1=1, max_norm2=0.25):
        super(EEGNetModel, self).__init__()
        linear_size = (time_points // (pk1 * pk2)) * f2

        # Temporal Filters
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, temp_kernel), padding='same', bias=False),
            nn.BatchNorm2d(f1),
        )

        # Spatial Filters
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, pk1)),
            nn.Dropout(dropout_rate)
        )

        # Separable Conv Filters
        self.block3 = nn.Sequential(
            nn.Conv2d(d * f1, f2, (1, 16), groups=f2, bias=False, padding='same'),
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pk2)),
            nn.Dropout(dropout_rate)
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(linear_size, classes)

        # Apply max_norm constraint to the depthwise layer in block2
        self._apply_max_norm(self.block2[0], max_norm1)
        self._apply_max_norm(self.fc, max_norm2)

    def _apply_max_norm(self, layer, max_norm):
        for name, param in layer.named_parameters():
            if 'weight' in name:
                param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=max_norm)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)  # Flatten
        # print(x.shape)
        # temp = x
        x = self.fc(x)
        # print(x.shape)
        return x
        # return temp, x
        # print(temp.shape)


class S6(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device):
        super(S6, self).__init__()
        # 线性变换
        self.fc1 = nn.Linear(d_model, d_model).to(device)
        self.fc2 = nn.Linear(d_model, state_size).to(device)
        self.fc3 = nn.Linear(d_model, state_size).to(device)

        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size


        self.A = nn.Parameter(F.normalize(torch.ones(d_model, state_size, device=device), p=2, dim=-1))  # A（D.N）
        nn.init.xavier_uniform_(self.A)

        self.B = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)  # B（B,l,n）
        self.C = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)  # C（B,l,n）

        self.delta = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)  # delta （B,l,D）

        self.dA = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)     # dA（B,l,D，N）
        self.dB = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)      # dB（B,l,D，N）

        self.h = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)   # h（B,l,D，N）
        self.y = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)    # y（B,L,D）


    def discretization(self):
        self.dB = torch.einsum("bld,bln->bldn", self.delta, self.B)
        self.dA = torch.exp(torch.einsum("bld,dn->bldn", self.delta, self.A))
        return self.dA, self.dB

    def forward(self, x):
        x = x.to(device)

        self.B = self.fc2(x)
        self.C = self.fc3(x)
        self.delta = F.softplus(self.fc1(x))

        self.discretization()

        if DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM:
            global current_batch_size
            current_batch_size = x.shape[0]

            if self.h.shape[0] != current_batch_size:
                different_batch_size = True
                h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h[:current_batch_size, ...]) + rearrange(x,
                                                                                                             "b l d -> b l d 1") * self.dB
            else:
                different_batch_size = False
                h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h) + rearrange(x, "b l d -> b l d 1") * self.dB

            # y [batch_size, seq_len, d_model]
            self.y = torch.einsum('bln,bldn->bld', self.C, h_new)

            global temp_buffer
            temp_buffer = h_new.detach().clone() if not self.h.requires_grad else h_new.clone()

            return self.y

        else:
            # h [batch_size, seq_len, d_model, state_size]
            h = torch.zeros(x.size(0), self.seq_len, self.d_model, self.state_size, device=x.device)
            y = torch.zeros_like(x)

            h = torch.einsum('bldn,bldn->bldn', self.dA, h) + rearrange(x, "b l d -> b l d 1") * self.dB

            # y [batch_size, seq_len, d_model]
            y = torch.einsum('bln,bldn->bld', self.C, h)

            return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 device: str = 'cuda'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class MambaBlock(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device):
        super(MambaBlock, self).__init__()

        self.inp_proj = nn.Linear(d_model, 2 * d_model).to(device)
        self.out_proj = nn.Linear(2 * d_model, d_model).to(device)

        # For residual skip connection
        self.D = nn.Linear(d_model, 2 * d_model).to(device)

        # Set _no_weight_decay attribute on bias
        self.out_proj.bias._no_weight_decay = True

        # Initialize bias to a small constant value
        nn.init.constant_(self.out_proj.bias, 1.0)

        self.S6 = S6(seq_len, 2 * d_model, state_size, device)

        # Add 1D convolution with kernel size 3
        self.conv = nn.Conv1d(seq_len, seq_len, kernel_size=3, padding=1).to(device)

        # Add linear layer for conv output
        self.conv_linear = nn.Linear(2 * d_model, 2 * d_model).to(device)

        # rmsnorm
        self.norm = RMSNorm(d_model, device=device)

    def forward(self, x):
        x = x.to(device)
        x = self.norm(x)

        # projection
        x_proj = self.inp_proj(x)

        # Add 1D convolution with kernel size 3
        x_conv = self.conv(x_proj)
        x_conv_act = F.silu(x_conv)

        # Add linear layer for conv output
        x_conv_out = self.conv_linear(x_conv_act)

        # s6
        x_ssm = self.S6(x_conv_out)
        x_act = F.silu(x_ssm)  # Swish activation can be implemented as x * sigmoid(x)

        # residual skip connection with nonlinearity introduced by multiplication
        x_residual = F.silu(self.D(x))

        x_combined = x_act * x_residual

        x_out = self.out_proj(x_combined)

        return x_out


class Mamba(nn.Module):
    def __init__(self, input_channels=22, seq_len=1000, d_model=16, state_size=32, output_dim=4, device=device):
        super(Mamba, self).__init__()
        self.device = device
        self.input_proj = nn.Linear(input_channels, d_model).to(device)  # Map input features to d_model
        self.mamba_block1 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block2 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block3 = MambaBlock(seq_len, d_model, state_size, device)
        self.output_proj = nn.Linear(d_model, output_dim).to(device)  # Map final output to target classes

    def forward(self, x):
        x = x.to(device)
        # print(x.shape)  # torch.Size([72, 224, 1000])
        # Input shape: [batch_size, feature_size, seq_len]
        # batch_size, _, feature_size, seq_len = x.shape
        batch_size, feature_size, seq_len = x.shape
        x = x.transpose(1, 2)
        # x = x.view(batch_size, seq_len, feature_size)  # Reshape to [batch_size, seq_len, feature_size]
        # print(x.shape)    # torch.Size([72, 1000, 224])

        # Project input features to d_model
        x = self.input_proj(x)  # Shape: [batch_size, seq_len, d_model]
        # print(self.input_proj(x).shape)

        # Process through Mamba blocks
        x = self.mamba_block1(x)
        x = self.mamba_block2(x)
        x = self.mamba_block3(x)

        # Pool over sequence length
        x = x.mean(dim=1)  # Global average pooling: Shape [batch_size, d_model]

        # Project to output dimension
        out = self.output_proj(x)  # Shape: [batch_size, output_dim]
        return x, out


class CNN_Mamba(nn.Module):
    def __init__(self, chans=22, time_points=1000, temp_kernel=32,
                 f1=16, f2=32, d=2, pk1=8, pk2=16, dropout_rate=0.5,
                 max_norm1=1, max_norm2=0.25, seq_len=1000, d_model=16,
                 state_size=32, output_dim=4, device='cuda'):
        super(CNN_Mamba, self).__init__()

        self.eegnet = EEGNetModel(chans, classes=output_dim, time_points=time_points, temp_kernel=temp_kernel,
                                  f1=f1, f2=f2, d=d, pk1=pk1, pk2=pk2, dropout_rate=dropout_rate,
                                  max_norm1=max_norm1, max_norm2=max_norm2)

        # self.mamba = Mamba(input_channels=f1 * d, seq_len=seq_len, d_model=d_model, state_size=state_size,
        #                    output_dim=output_dim, device=device)
        self.mamba = Mamba(input_channels=4, seq_len=seq_len, d_model=d_model, state_size=state_size,
                           output_dim=output_dim, device=device)     # TODO

        self.seq_len = seq_len  #

    def forward(self, x):
        eegnet_output = self.eegnet(x)  # eegnet_output 的形状是 [batch_size, output_dim]
        temp = eegnet_output.unsqueeze(1).repeat(1, self.seq_len, 1)
        # print(temp.shape)
        temp = temp.transpose(1, 2)
        # print(temp.shape)  # torch.Size([72, 4, 1000])
        mamba_output, final_output = self.mamba(temp)

        return mamba_output, final_output


if __name__ == '__main__':
    x = torch.randn(72, 1, 22, 1000).cuda()
    model = CNN_Mamba().cuda()
    y = model(x)
    for i in y:
        print(i.shape)



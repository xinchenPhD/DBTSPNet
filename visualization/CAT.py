# import argparse
# import os
# # gpus = [1]
# # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
# # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
# import math
# import glob
# import random
# import itertools
# import datetime
# import time
# import datetime
# import sys
# import scipy.io

# import torchvision.transforms as transforms
# from torchvision.utils import save_image, make_grid
#
# from torch.utils.data import DataLoader
# from torch.autograd import Variable
# from torchsummary import summary
# import torch.autograd as autograd
# from torchvision.models import vgg19

# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# import torch.nn.init as init

# from torch.utils.data import Dataset
# from PIL import Image
# import torchvision.transforms as transforms
# from sklearn.decomposition import PCA

import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt

from torch import nn
# from torch import Tensor
# from PIL import Image
# from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
# from einops.layers.torch import Rearrange, Reduce
# # from common_spatial_pattern import csp
from scipy.io import loadmat

import matplotlib.pyplot as plt
from torch.backends import cudnn
# from grad_cam.utils import GradCAM, show_cam_on_image
# from utils import GradCAM, show_cam_on_image
from visualization.utils import GradCAM

cudnn.benchmark = False
cudnn.deterministic = True


mat = loadmat(r'E:\model_updating-------\EEGNet\EEG\standard_BCICIV_2a_data\A09E.mat')
data = mat['data']  # data - (samples, channels, trials)   [1000,22,288]
# print(data.shape)
label = mat['label']  # label -  (label, 1)
# print(label.shape)


def reshape_transform(tensor):
    result = rearrange(tensor, 'b (h w) e -> b e (h) (w)', h=1)
    return result


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Conformer().cuda()
model.load_state_dict(torch.load(r'E:\model_updating-------\EEGNet\EEG\Results\Conformer_state_each_sub\subject_9_model.pth',
                                 map_location=device))
target_layers = [model[1]]  # set the target layer  某一层
# target_layers = [model[0].projection]
# target_layers = [model[0]]  # set the layer you want to visualize, you can use torchsummary here to find the layer index
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True, reshape_transform=reshape_transform)
# cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)


# TODO: Class Activation Topography (proposed in the paper)
import mne
from matplotlib import mlab as mlab

biosemi_montage = mne.channels.make_standard_montage('biosemi64')
index = [37, 9, 10, 46, 45, 44, 13, 12, 11, 47, 48, 49, 50, 17, 18, 31, 55, 54, 19, 30, 56, 29]  # 22个通道
biosemi_montage.ch_names = [biosemi_montage.ch_names[i] for i in index]
biosemi_montage.dig = [biosemi_montage.dig[i + 3] for i in index]
info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=250., ch_types='eeg')

all_cam = []
# this loop is used to obtain the cam of each trial/sample   288个trail
data = torch.as_tensor(data, dtype=torch.float32)
data = data.permute(2, 1, 0)
data = data.unsqueeze(1)
# print(data.shape)  # 288, 1, 22, 1000
label = np.squeeze(np.transpose(label))  # (288,)
# print(label.shape)

idx = np.where(label == 4)
data = data[idx]
# print(data.shape)
for i in range(72):
    test = torch.as_tensor(data[i:i + 1, :, :, :], dtype=torch.float32)
    test = torch.autograd.Variable(test, requires_grad=True)
    grayscale_cam = cam(input_tensor=test)
    grayscale_cam = grayscale_cam[0, :]
    all_cam.append(grayscale_cam)

data = data.numpy()
test_all_data = np.squeeze(np.mean(data, axis=0))
test_all_data = (test_all_data - np.mean(test_all_data)) / np.std(test_all_data)
mean_all_test = np.mean(test_all_data, axis=1)

# the mean of all cam
test_all_cam = np.mean(all_cam, axis=0)
test_all_cam = (test_all_cam - np.mean(test_all_cam)) / np.std(test_all_cam)
mean_all_cam = np.mean(test_all_cam, axis=1)

# apply cam on the input data
hyb_all = test_all_data * test_all_cam
hyb_all = (hyb_all - np.mean(hyb_all)) / np.std(hyb_all)
mean_hyb_all = np.mean(hyb_all, axis=1)


evoked = mne.EvokedArray(test_all_data, info)
evoked.set_montage(biosemi_montage)

plt.figure(1)
im, cn = mne.viz.plot_topomap(mean_hyb_all, evoked.info, show=False)
plt.colorbar(im)
plt.show()


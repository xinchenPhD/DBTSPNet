import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepConvNet(nn.Module):
    def __init__(self, nb_classes=4, Chans=22, dropoutRate=0.5,max_norm2=0.5):
        super(DeepConvNet, self).__init__()

        # Block 1
        self.conv1_1 = nn.Conv2d(1, 25, kernel_size=(1, 5), bias=False)  # Input shape: [batch, 1, Chans, Samples]
        self.conv1_2 = nn.Conv2d(25, 25, kernel_size=(Chans, 1), bias=False)  # Spatial convolution
        # self.batchnorm1 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.9)
        self.batchnorm1 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.9)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.dropout1 = nn.Dropout(dropoutRate)

        # Block 2
        self.conv2 = nn.Conv2d(25, 50, kernel_size=(1, 5), bias=False)
        # self.batchnorm2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.9)
        self.batchnorm2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.9)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.dropout2 = nn.Dropout(dropoutRate)

        # Block 3
        self.conv3 = nn.Conv2d(50, 100, kernel_size=(1, 5), bias=False)
        # self.batchnorm3 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.9)
        self.batchnorm3 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.9)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.dropout3 = nn.Dropout(dropoutRate)

        # Block 4
        self.conv4 = nn.Conv2d(100, 200, kernel_size=(1, 5), bias=False)
        # self.batchnorm4 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.9)
        self.batchnorm4 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.9)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.dropout4 = nn.Dropout(dropoutRate)

        # Fully connected layer
        self.fc = nn.Linear(11600, nb_classes)  # Adjust for pooling layers
        # self.fc = nn.Linear(224, nb_classes)  # Adjust for pooling layers
        # self._apply_max_norm(self.fc, max_norm2)
        self.softmax = nn.Softmax(dim=1)
        # self._apply_max_norm(self.fc, max_norm2)

    # def _apply_max_norm(self, layer, max_norm):
    #     for name, param in layer.named_parameters():
    #         if 'weight' in name:
    #             param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=max_norm)

    def forward(self, x):
        # Block 1
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Block 4
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = F.elu(x)
        x = self.pool4(x)
        x = self.dropout4(x)

        # Flatten and Fully Connected Layer
        x = torch.flatten(x, start_dim=1)
        # temp = x
        x = self.fc(x)
        temp = x
        x = self.softmax(x)
        return temp, x


if __name__ == '__main__':
    x = torch.randn(72, 1, 22, 1000).cuda()
    model = DeepConvNet().cuda()
    y = model(x)
    # print(y.shape)
    for i in y:
        print(i.shape)
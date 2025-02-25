import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    def __init__(self, number_channel=22, nb_classes=4, dropout_rate=0.5):
        super(CNN_LSTM, self).__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (number_channel, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(dropout_rate),
        )

        self.lstm_input_size = 40
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
        )

        self.fc = nn.Linear(128, nb_classes)

    def forward(self, x):
        x = self.shallownet(x)
        x = x.view(x.size(0), -1, 40)

        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]

        output = self.fc(lstm_out)

        temp = output

        return temp, output


if __name__ == '__main__':
    x = torch.randn(72, 1, 22, 1000).cuda()
    model = CNN_LSTM().cuda()
    y = model(x)
    for i in y:
        print(i.shape)

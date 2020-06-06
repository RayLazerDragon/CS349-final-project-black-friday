import torch
from torch.nn import Module, Sequential
from torch.nn import Linear, LeakyReLU, ReLU, BatchNorm1d
from torchsummary import summary
from torch.utils.data.dataset import Dataset


class MyDataSet(Dataset):

    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item], self.target[item]


class BlackFriday(Module):

    def __init__(self):
        super(BlackFriday, self).__init__()

        self.layer1 = Sequential(
            Linear(11, 24),
            ReLU(),
            BatchNorm1d(24)
        )
        self.layer2 = Sequential(
            Linear(24, 48),
            ReLU(),
            BatchNorm1d(48)
        )

        self.layer3 = Sequential(
            Linear(48, 72),
            ReLU(),
            BatchNorm1d(72)
        )

        self.layer4 = Sequential(
            Linear(72, 128),
            ReLU(),
            BatchNorm1d(128)
        )

        self.layer5 = Sequential(
            Linear(128, 72),
            ReLU(),
            BatchNorm1d(72)
        )

        self.layer6 = Sequential(
            Linear(72, 60),
            ReLU(),
            BatchNorm1d(60)
        )

        self.layer7 = Sequential(
            Linear(60, 30),
            ReLU(),
            BatchNorm1d(30)
        )

        self.layer8 = Sequential(
            Linear(30, 10),
            ReLU(),
            BatchNorm1d(10)
        )

        self.output = Linear(10, 1)

    def forward(self, data):
        out = self.layer1(data)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.output(out)

        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BlackFriday().to(device)

    summary(model, input_size=(11,))

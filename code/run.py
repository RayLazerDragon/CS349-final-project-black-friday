from code.model import *
from code.load_data import load_data
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
import torch
import numpy as np
import matplotlib.pyplot as plt


def normalize(train_data, train_target, valid_data, valid_target):
    target_mean = np.mean(train_target)
    target_std = np.std(train_target)

    # normalize the output
    train_target = (train_target - target_mean) / target_std
    valid_target = (valid_target - target_mean) / target_std

    # normalize age
    age = train_data[:, 1]
    age_mean = np.mean(age)
    age_std = np.std(age)
    train_data[:, 1] = (age - age_mean) / age_std
    valid_data[:, 1] = (valid_data[:, 1] - age_mean) / age_std

    # normalize occupation
    occupation = train_data[:, 2]
    occ_mean, occ_std = np.mean(occupation), np.std(occupation)
    train_data[:, 2] = (train_data[:, 2] - occ_mean) / occ_std
    valid_data[:, 2] = (valid_data[:, 2] - occ_mean) / occ_std

    # normalize stay in current city years
    yrs = train_data[:, 3]
    yrs_mean, yrs_std = np.mean(yrs), np.std(yrs)
    train_data[:, 3] = (train_data[:, 3] - yrs_mean) / yrs_std
    valid_data[:, 3] = (valid_data[:, 3] - yrs_mean) / yrs_std

    # normalize category 1 2 3
    c1 = train_data[:, 5]
    c1_mean, c1_std = np.mean(c1), np.std(c1)
    train_data[:, 5] = (train_data[:, 5] - c1_mean) / c1_std
    valid_data[:, 5] = (valid_data[:, 5] - c1_mean) / c1_std

    c2 = train_data[:, 6]
    c2_mean, c2_std = np.mean(c2), np.std(c2)
    train_data[:, 6] = (train_data[:, 6] - c2_mean) / c2_std
    valid_data[:, 6] = (valid_data[:, 6] - c2_mean) / c2_std

    c3 = train_data[:, 7]
    c3_mean, c3_std = np.mean(c3), np.std(c3)
    train_data[:, 7] = (train_data[:, 7] - c3_mean) / c3_std
    valid_data[:, 7] = (valid_data[:, 7] - c3_mean) / c3_std

    return train_data, train_target, valid_data, valid_target


def run_model(model, train_set, valid_set, batch_size=512, num_epoch=5000, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_loss = []
    valid_loss = []

    model.double()
    for n in range(num_epoch):

        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if n % 10 == 0:
            tr_loss = evaluate(model, train_loader)
            train_loss.append(tr_loss)

            val_loss = evaluate(model, valid_loader)
            valid_loss.append(val_loss)

            print(f'Training loss is {tr_loss}, valid loss is {val_loss} in epoch {n}.')

    return model, train_loss, valid_loss


def evaluate(model, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = MSELoss(reduction='sum')
    model.eval()

    with torch.no_grad():
        
        loss = 0
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()

    return loss / len(data_loader.dataset)


def visualize(train_loss, valid_loss):
    x = list(range(0, len(train_loss) * 10, 10))
    plt.plot(x, train_loss, label='training loss')
    plt.plot(x, valid_loss, label='valid loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'The model is running on {device}')

    train_data, train_target, valid_data, valid_target = load_data()

    train_data, train_target, valid_data, valid_target = normalize(train_data, train_target, valid_data, valid_target)

    train_set = MyDataSet(train_data, train_target)
    valid_set = MyDataSet(valid_data, valid_target)

    model = BlackFriday().to(device)
    summary(model, (11,))

    # model, train_loss, valid_loss = run_model(model, train_set, valid_set, num_epoch=100, batch_size=1024,
    #                                           learning_rate=1e-4)
    # visualize(train_loss, valid_loss)
    #
    # torch.save(model.state_dict(), '../results/BlackFriday.pth')

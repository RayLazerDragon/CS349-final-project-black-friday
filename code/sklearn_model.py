import numpy as np
from sklearn.ensemble import RandomForestRegressor
from code.load_data import load_data
from code.run import normalize


def RandomForest(train_data, train_target, valid_data, valid_target):

    train_target = np.reshape(train_target, (train_target.shape[0], ))
    valid_target = np.reshape(valid_target, (valid_target.shape[0],))

    model = RandomForestRegressor(max_depth=20, min_samples_leaf=128)

    model.fit(train_data, train_target)
    valid_predict = model.predict(valid_data)
    train_predict = model.predict(train_data)

    train_loss = np.sum((train_predict - train_target) ** 2) / train_target.shape[0]
    print(f'train loss is {train_loss}')

    valid_loss = np.sum((valid_predict-valid_target)**2) / valid_target.shape[0]
    print(f'valid loss is {valid_loss}')

    return train_loss, valid_loss


if __name__ == '__main__':
    train_data, train_target, valid_data, valid_target = load_data()
    train_data, train_target, valid_data, valid_target = normalize(train_data, train_target, valid_data, valid_target)
    train_loss, valid_loss = RandomForest(train_data, train_target, valid_data, valid_target)




import numpy as np
import pandas as pd
import math


def load_csv():
    path = f'../data/train.csv'

    data = np.array(pd.read_csv(path))

    train_data = data[:, 2:11]
    train_target = data[:, -1]

    return train_data, train_target


def split_data(data, target):
    # print(target[400000:])
    return data[:400000], target[:400000], data[400000:], target[400000:]


def preprocessing(data, target, dataset='train'):
    # print('before preprocessing')
    # print(data[2:6])
    # map gender to 0 or 1
    data = np.where(data == 'F', 0, data)
    data = np.where(data == 'M', 1, data)

    # map age from a range to integer
    data = np.where(data == '0-17', 1, data)
    data = np.where(data == '18-25', 2, data)
    data = np.where(data == '26-35', 3, data)
    data = np.where(data == '36-45', 4, data)
    data = np.where(data == '46-50', 5, data)
    data = np.where(data == '51-55', 6, data)
    data = np.where(data == '55+', 7, data)

    # stay in current city years
    data = np.where(data == '0', 0, data)
    data = np.where(data == '1', 1, data)
    data = np.where(data == '2', 2, data)
    data = np.where(data == '3', 3, data)
    data = np.where(data == '4+', 4, data)

    data[pd.isnull(data)] = 0

    result = []

    for d in data:
        # print(data)
        if 'A' in d:
            res = np.delete(d, obj=3, axis=0)
            res = np.append(res, 1)
            res = np.append(res, 0)
            res = np.append(res, 0)
            result.append(res)
        elif 'B' in d:
            res = np.delete(d, obj=3, axis=0)
            res = np.append(res, 0)
            res = np.append(res, 1)
            res = np.append(res, 0)
            result.append(res)
        elif 'C' in d:
            res = np.delete(d, obj=3, axis=0)
            res = np.append(res, 0)
            res = np.append(res, 0)
            res = np.append(res, 1)
            result.append(res)
    del data

    # divide target by 10000

    # target /= 10000.0
    col_names = ['Gender', 'Age', 'Occupation', 'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',
                 'Product_Category_2', 'Product_Category_3', 'from_A', 'from_B', 'from_C']

    result = pd.DataFrame(result)
    result.columns = col_names
    # print(result)
    result.to_csv(f'../data/processed_{dataset}_data.csv')

    pd.DataFrame(target).to_csv(f'../data/processed_{dataset}_target.csv')


def load_data():  # load preprocessed data
    train_data = np.array(pd.read_csv('../data/processed_train_data.csv'))
    train_target = np.array(pd.read_csv('../data/processed_train_target.csv'))
    valid_data = np.array(pd.read_csv('../data/processed_valid_data.csv'))
    valid_target = np.array(pd.read_csv('../data/processed_valid_target.csv'))

    return train_data[:, 1:], train_target[:, 1:], valid_data[:, 1:], valid_target[:, 1:]


if __name__ == '__main__':
    # train_data, train_target = load_csv()

    # train_data, train_target, valid_data, valid_target = split_data(train_data, train_target)

    # preprocessing(train_data, train_target, 'train')

    # preprocessing(valid_data, valid_target, 'valid')
    train_data, train_target, valid_data, valid_target = load_data()

    print(train_data.shape)
    print(train_target.shape)

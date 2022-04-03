import pandas as pd
import numpy as np


def dataset_1():
    dfs = pd.read_excel('data/данные.xlsx', sheet_name='Лист1')
    dataset = {
        'input': dfs['норм егэ'].to_numpy(),
        'output': [
            dfs['Норм дифф исчисление'].to_numpy(),
            dfs['Норм дискретная матем'].to_numpy(),
            dfs['Норм алгебра'].to_numpy(),
            dfs['Норм основы прогр'].to_numpy()
        ]
    }
    x = np.array(dataset['input']).transpose()
    y = np.array(dataset['output']).transpose()
    return x, y


def dataset_2():
    dfs = pd.read_excel('data/данные.xlsx', sheet_name='Лист1')
    dataset = {
        'input': dfs['норм егэ'].to_numpy(),
        'output': [
            dfs['Норм дифф исчисление 2сем'].to_numpy(),
            dfs['Норм дискретная матем 2 сем'].to_numpy(),
            dfs['Норм Алгебра 2 сем'].to_numpy(),
            dfs['Норм основы прогр 2 сем'].to_numpy()
        ]
    }
    x = np.array(dataset['input']).transpose()
    y = np.array(dataset['output']).transpose()
    return x, y


def dataset_3():
    dfs = pd.read_excel('data/данные.xlsx', sheet_name='Лист1')
    dataset = {
        'input': [
            dfs['Норм дифф исчисление'].to_numpy(),
            dfs['Норм дискретная матем'].to_numpy(),
            dfs['Норм алгебра'].to_numpy(),
            dfs['Норм основы прогр'].to_numpy()
        ],
        'output': [
            dfs['Норм дифф исчисление 2сем'].to_numpy(),
            dfs['Норм дискретная матем 2 сем'].to_numpy(),
            dfs['Норм Алгебра 2 сем'].to_numpy(),
            dfs['Норм основы прогр 2 сем'].to_numpy()
        ]
    }
    x = np.array(dataset['input']).transpose()
    y = np.array(dataset['output']).transpose()
    x = np.array([sum(line) / 4 for line in x])
    print(f"x_shape == {x.shape}\ny_shape == {y.shape}")
    return x, y


def dataset_4():
    dfs = pd.read_excel('data/данные.xlsx', sheet_name='Лист1')
    dataset = {
        'input_1': dfs['норм егэ'].to_numpy(),
        'input_2': [
            dfs['Норм дифф исчисление'].to_numpy(),
            dfs['Норм дискретная матем'].to_numpy(),
            dfs['Норм алгебра'].to_numpy(),
            dfs['Норм основы прогр'].to_numpy()
        ],
        'output': [
            dfs['Норм дифф исчисление 2сем'].to_numpy(),
            dfs['Норм дискретная матем 2 сем'].to_numpy(),
            dfs['Норм Алгебра 2 сем'].to_numpy(),
            dfs['Норм основы прогр 2 сем'].to_numpy()
        ]
    }
    x1 = np.array(dataset['input_1']).transpose()
    x2 = np.array(dataset['input_2']).transpose()
    y = np.array(dataset['output']).transpose()
    x2 = np.array([sum(line) / 4 for line in x2])
    x = np.array([[i1, i2] for i1, i2 in zip(x1, x2)])
    print(f"x_shape == {x.shape}\ny_shape == {y.shape}")
    return x, y


def dataset_5():
    dfs = pd.read_excel('data/данные.xlsx', sheet_name='Лист1')
    dataset = {
        'input_1': dfs['норм егэ'].to_numpy(),
        'input_2': dfs['пол'].to_numpy(),
        'input_3': dfs['возраст'].to_numpy(),
        'output': [
            dfs['Норм дифф исчисление'].to_numpy(),
            dfs['Норм дискретная матем'].to_numpy(),
            dfs['Норм алгебра'].to_numpy(),
            dfs['Норм основы прогр'].to_numpy()
        ]
    }
    x1 = np.array(dataset['input_1']).transpose()
    x2 = np.array(dataset['input_2']).transpose()
    x2 = np.array([0. if a == 'ж' else 1. for a in x2])
    x3 = np.array(dataset['input_3']).transpose()
    x3 = np.array([(a - min(x3)) / (max(x3) - min(x3)) for a in x3])
    y = np.array(dataset['output']).transpose()
    x = np.array([[i1, i2, i3] for i1, i2, i3 in zip(x1, x2, x3)])
    print(f"x_shape == {x.shape}\ny_shape == {y.shape}")
    return x, y


def dataset_6():
    dfs = pd.read_excel('data/данные.xlsx', sheet_name='Лист1')
    dataset = {
        'input': dfs['норм егэ'].to_numpy(),
        'output': [
            dfs['Норм дифф исчисление'].to_numpy(),
            dfs['Норм дискретная матем'].to_numpy(),
            dfs['Норм алгебра'].to_numpy(),
            dfs['Норм основы прогр'].to_numpy()
        ]
    }
    x = np.array(dataset['input']).transpose()
    y = np.array(dataset['output']).transpose()
    y = np.array([sum(line) / 4 for line in y])
    print(f"x_shape == {x.shape}\ny_shape == {y.shape}")
    return x, y


if __name__ == "__main__":
    pass
    # dataset_5()
    dataset_6()

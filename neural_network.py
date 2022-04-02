from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def show_plot(y_pred, y_exp):
    x = range(len(y_pred))

    plt.subplot(2, 2, 1)
    plt.scatter(x, [line[0] for line in y_pred], s=30, alpha=0.5, label="Прогноз")
    plt.scatter(x, [line[0] for line in y_exp], s=30, alpha=0.5, label="Ожидание")
    plt.title('Дифференциальные исчисления')

    plt.subplot(2, 2, 2)
    plt.scatter(x, [line[1] for line in y_pred], s=30, alpha=0.5, label="Прогноз")
    plt.scatter(x, [line[1] for line in y_exp], s=30, alpha=0.5, label="Ожидание")
    plt.title('Дискретная математика')

    plt.subplot(2, 2, 3)
    plt.scatter(x, [line[2] for line in y_pred], s=30, alpha=0.5, label="Прогноз")
    plt.scatter(x, [line[2] for line in y_exp], s=30, alpha=0.5, label="Ожидание")
    plt.title('Алгебра')

    plt.subplot(2, 2, 4)
    plt.scatter(x, [line[3] for line in y_pred], s=30, alpha=0.5, label="Прогноз")
    plt.scatter(x, [line[3] for line in y_exp], s=30, alpha=0.5, label="Ожидание")
    plt.title('Основы программирования')

    plt.get_current_fig_manager().window.showMaximized()
    plt.show()


def check_model_accuracy(m: Sequential, x_in, y_out):
    predict = m.predict(x_in)
    res = 0
    y_pred, y_exp = [], []
    for pr_line, ex_line in zip(predict, y_out):
        pr_line = [round(a * 3 + 2) for a in pr_line]
        y_pred.append(pr_line)
        ex_line = [round(a * 3 + 2) for a in ex_line]
        y_exp.append(ex_line)
        res += sum(a == b for a, b in zip(pr_line, ex_line))

    res = 100 * res / (len(y_out) * len(y_out[0]))
    print(f"result (%) == {res}%")
    show_plot(y_pred, y_exp)

    return res


################################
# Исследование № 1
# Входные данные:
#     1 Баллы ЕГЭ
# Выходные данные (1-й семестр):
#     1 дифф исчисление
#     2 дискретная матем
#     3 алгебра
#     4 основы прогр
################################
def prediction_1():
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=1))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=4, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

    from data import dataset_1
    x, y = dataset_1()

    model.fit(x=x, y=y, epochs=20000)

    result = check_model_accuracy(model, x, y)

    model.save(filepath=f'models/predictions_1/model_{result:.4f}.h5')


def prediction_1_test():
    model = load_model("models/predictions_1/model_65.3226.h5")

    from data import dataset_1
    x, y = dataset_1()

    check_model_accuracy(model, x, y)


################################
# Исследование № 2
# Модель обучена на данных 1 - го семестра
# Входные данные:
#     1 Баллы ЕГЭ
# Выходные данные (2-й семестр):
#     1 дифф исчисление
#     2 дискретная матем
#     3 алгебра
#     4 основы прогр
################################
def prediction_2():
    model = load_model("models/predictions_1/model_65.3226.h5")

    from data import dataset_2
    x, y = dataset_2()

    check_model_accuracy(model, x, y)


################################
# Исследование № 3
# Модель обучена на данных 1 - го семестра
# Входные данные:
#     1 Сумма баллов по обучению за 1-й семестр
# Выходные данные (2-й семестр):
#     1 дифф исчисление
#     2 дискретная матем
#     3 алгебра
#     4 основы прогр
################################
def prediction_3():
    model = load_model("models/predictions_1/model_65.3226.h5")

    from data import dataset_3
    x, y = dataset_3()

    check_model_accuracy(model, x, y)


################################
# Исследование № 4
# Входные данные:
#     1 Баллы ЕГЭ
#     2 Сумма баллов по обучению за 1-й семестр
# Выходные данные (2-й семестр):
#     1 дифф исчисление
#     2 дискретная матем
#     3 алгебра
#     4 основы прогр
################################
def prediction_4():
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=2))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=4, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

    from data import dataset_4
    x, y = dataset_4()

    model.fit(x=x, y=y, epochs=20000)

    result = check_model_accuracy(model, x, y)

    model.save(filepath=f'models/predictions_4/model_{result:.4f}.h5')


def prediction_4_test():
    model = load_model("models/predictions_4/model_74.1935.h5")

    from data import dataset_4
    x, y = dataset_4()

    check_model_accuracy(model, x, y)


################################
# Исследование № 5
# Входные данные:
#     1 Баллы ЕГЭ
#     2 Пол
#     3 Возраст
# Выходные данные (1-й семестр):
#     1 дифф исчисление
#     2 дискретная матем
#     3 алгебра
#     4 основы прогр
################################
def prediction_5():
    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_dim=3))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=4, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

    from data import dataset_5
    x, y = dataset_5()

    model.fit(x=x, y=y, epochs=20000)

    result = check_model_accuracy(model, x, y)

    model.save(filepath=f'models/predictions_5/model_{result:.4f}.h5')


def prediction_5_test():
    model = load_model("models/predictions_5/model_86.6935.h5")

    from data import dataset_5
    x, y = dataset_5()

    check_model_accuracy(model, x, y)


if __name__ == "__main__":
    pass
    # prediction_1()
    # prediction_1_test()
    # prediction_2()
    # prediction_3()
    # prediction_4()
    # prediction_4_test()
    # prediction_5()
    prediction_5_test()

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def show_plot(y_pred, y_exp, predict_mode):
    if predict_mode == 4:
        x = range(len(y_pred))

        plt.subplot(2, 2, 1)
        plt.scatter(x, [line[0] for line in y_pred], s=30, alpha=0.7, c='r', label="Спрогнозировано")
        plt.scatter(x, [line[0] for line in y_exp], s=30, alpha=0.7, c='g', label="Ожидалось")
        plt.plot([line[0] for line in y_pred], c='r')
        plt.plot([line[0] for line in y_exp], c='g')
        plt.title('Дифференциальные исчисления')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.scatter(x, [line[1] for line in y_pred], s=30, alpha=0.7, c='r', label="Спрогнозировано")
        plt.scatter(x, [line[1] for line in y_exp], s=30, alpha=0.7, c='g', label="Ожидалось")
        plt.plot([line[1] for line in y_pred], c='r')
        plt.plot([line[1] for line in y_exp], c='g')
        plt.title('Дискретная математика')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.scatter(x, [line[2] for line in y_pred], s=30, alpha=0.7, c='r', label="Спрогнозировано")
        plt.scatter(x, [line[2] for line in y_exp], s=30, alpha=0.7, c='g', label="Ожидалось")
        plt.plot([line[2] for line in y_pred], c='r')
        plt.plot([line[2] for line in y_exp], c='g')
        plt.title('Алгебра')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.scatter(x, [line[3] for line in y_pred], s=30, alpha=0.7, c='r', label="Спрогнозировано")
        plt.scatter(x, [line[3] for line in y_exp], s=30, alpha=0.7, c='g', label="Ожидалось")
        plt.plot([line[3] for line in y_pred], c='r')
        plt.plot([line[3] for line in y_exp], c='g')
        plt.title('Основы программирования')
        plt.legend()

        plt.get_current_fig_manager().window.showMaximized()
        plt.show()
    elif predict_mode == 1:
        x = range(len(y_pred))
        plt.scatter(x, y_pred, s=60, alpha=0.8, c='r', label="Спрогнозировано")
        plt.scatter(x, y_exp, s=60, alpha=0.8, c='g', label="Ожидалось")
        plt.plot(y_pred, c='r')
        plt.plot(y_exp, c='g')
        plt.legend()

        plt.get_current_fig_manager().window.showMaximized()
        plt.show()


def check_model_accuracy(m: Sequential, x_in, y_out, predict_mode=4):
    predict = m.predict(x_in)
    res = 0
    y_pred, y_exp = [], []
    if predict_mode == 4:
        for pr_line, ex_line in zip(predict, y_out):
            pr_line = [round(a * 3 + 2) for a in pr_line]
            y_pred.append(pr_line)
            ex_line = [round(a * 3 + 2) for a in ex_line]
            y_exp.append(ex_line)
            res += sum(a == b for a, b in zip(pr_line, ex_line))
        res = 100 * res / (len(y_out) * len(y_out[0]))
    elif predict_mode == 1:
        for pr_el, ex_el in zip(predict, y_out):
            pr_el = round(pr_el[0] * 4 * 3 + 8, 2)
            y_pred.append(pr_el / 4)
            ex_el = round(ex_el * 4 * 3 + 8, 2)
            y_exp.append(ex_el / 4)
            res += int(pr_el == ex_el)
        res = 100 * res / len(y_out)

    print(f"result (%) == {res}%")
    show_plot(y_pred, y_exp, predict_mode=predict_mode)

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


################################
# Исследование № 6
# Входные данные:
#     1 Баллы ЕГЭ
# Выходные данные (1-й семестр):
#     1 средний балл по всем предметам (1-й семестр)
################################
def prediction_6():
    model = Sequential()
    model.add(Dense(units=16, activation='relu', input_dim=1))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

    from data import dataset_6
    x, y = dataset_6()

    model.fit(x=x, y=y, epochs=20000)

    result = check_model_accuracy(model, x, y, predict_mode=1)

    model.save(filepath=f'models/predictions_6/model_{result:.4f}.h5')


def prediction_6_test():
    model = load_model("models/predictions_6/model_4.8387.h5")

    from data import dataset_6
    x, y = dataset_6()

    check_model_accuracy(model, x, y, predict_mode=1)


if __name__ == "__main__":
    pass
    # prediction_1()
    # prediction_1_test()
    # prediction_2()
    # prediction_3()
    # prediction_4()
    # prediction_4_test()
    # prediction_5()
    # prediction_5_test()
    prediction_6()
    # prediction_6_test()

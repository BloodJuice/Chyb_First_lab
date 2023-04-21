import math

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
import matplotlib.pyplot as plt



def ThirdPoint():

    yFhirdPoint = []
    yHkx = [[], [], [], [], []]
    Ymiddle = np.zeros(5)
    Y_middle_tk = np.zeros(5)
    tetta_middle_one = 0
    tetta_middle_two = 0


    # Расчитываем тетты для входного сигнала (3 пункт лр)
    for i_ in range(5):
        tetta_all[i_] = Minimize()

    # Расчёт Тетты средней №1 и №2
    for i_ in range(5):
        tetta_middle_one += tetta_all[i_][0] / 5.0
        tetta_middle_two += tetta_all[i_][1] / 5.0
    tetta_middle = np.array([tetta_middle_one, tetta_middle_two])

    # Расчёт Y
    for i_ in range(5):
        yFhirdPoint.append(y(tetta_else))

    # Расчёт Ycp
    for i_ in range(5):
        for j_ in range(N):
            Ymiddle[i_] += yFhirdPoint[i_][j_]
        Ymiddle[i_] /= N

    # Расчёт Y(tk + 1| tk + 1) и Yср(tk + 1| tk + 1)
    for i_ in range(5):
        for j_ in range(N):
            yHkx[i_].append(yFhirdPoint[i_][j_] - v_noise[j_])

        for j_ in range(N):
            Y_middle_tk[i_] += yHkx[i_][j_]
        Y_middle_tk[i_] /= N


    # Расчёт относительных ошибок оценивания:
    # delta_Tetta
    tetta_middle = tetta_true - tetta_middle
    tetta_sum1 = np.sqrt(pow(tetta_middle[0], 2) + pow(tetta_middle[1], 2))
    tetta_sum2 = np.sqrt(pow(tetta_true[0], 2) + pow(tetta_true[1], 2))
    print("Delta_tetta =", tetta_sum1 / tetta_sum2)

    # delta_Y
    Y_middle_tk = Ymiddle - Y_middle_tk
    tetta_sum1 = np.sqrt(pow(Y_middle_tk[0], 2) + pow(Y_middle_tk[1], 2) + pow(Y_middle_tk[2], 2) + pow(Y_middle_tk[3], 2) + pow(Y_middle_tk[4], 2))
    tetta_sum2 = np.sqrt(pow(Ymiddle[0], 2) + pow(Ymiddle[1], 2) + pow(Ymiddle[2], 2) + pow(Ymiddle[3], 2) + pow(Ymiddle[4], 2))
    print("Delta_Y =", tetta_sum1 / tetta_sum2)


def Minimize():
    res = []
    ############__2__##########
    lim = [-2.0, 0.01, -0.05, 1.5]
    bounds = Bounds([lim[0], lim[1]],  # [min x0, min x1]
                    [lim[2], lim[3]])  # [max x0, max x1]
    x_start = tetta_else
    result = minimize(Xi, x_start, method='cobyla')
    res.append(minimize(Xi, x_start, method='SLSQP', jac=dXi, bounds=bounds))
    # print("Тетты для нулевого порядка:", result.__getitem__("x"))
    # print("Тетты для первого порядка:", res)
    return result.__getitem__("x")

def graf3D():
    # Выводим трёхмерное полотно
    x_ = np.linspace(-2, -0.5, 100)
    y_ = np.linspace(0.01, 1.5, 100)

    xgrid, ygrid = np.meshgrid(x_, y_)
    zgrid = np.zeros((len(xgrid), len(ygrid)))

    min_ = 0
    mini = 0
    minj = 0
    max_ = 0

    for i in range(len(xgrid)):
        for j in range(len(ygrid)):
            zgrid[i][j] = Xi([x_[i], y_[j]])
            if zgrid[i][j] < min_:
                min_ = zgrid[i][j]
                mini = i
                minj = j
            if zgrid[i][j] > max_:
                max_ = zgrid[i][j]
    # Поиск минимального значения на графике для первого порядка
    print("Минимальное значение на графике: =", min_)
    print("Максимальное значение на графике: =", max_)
    print("Тетта1, соответствующая минимальному значению:", x_[mini])
    print("Тетта2, соответствующая минимальному значению:", y_[minj])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(ygrid, xgrid, zgrid)
    ax.set_xlabel('tetta1')
    ax.set_ylabel('tetta2')
    ax.set_zlabel('Xi')
    plt.show()

def y(tetta):
    F = np.array([[-0.8, 1.0], [tetta[0], 0]])
    Psi = np.array([tetta[1], 1.0])
    H = np.array([1.0, 0])
    x_t0 = np.array([0, 0])
    x_tk = []
    y_tk_plus_one = []
    l = k
    while l < N:
        if l == 0:
            x_tk = x_t0

        x_tk_plus_one = np.matmul(F, x_tk) + np.dot(Psi, u_t0)
        y_tk_plus_one.append(np.matmul(H, x_tk_plus_one) + v_noise[k])
        x_tk = x_tk_plus_one
        l += 1

    return y_tk_plus_one

def Xi(x):

    # Инициализация матриц/векторов:
    x_t0 = np.array([0 , 0])
    F = np.array([[-0.8, 1.0], [x[0], 0]])
    Psi = np.array([x[1], 1.0])
    H = np.array([1, 0])
    x_tk = []

    # Calculate Xi constant
    xi = N * m * v * math.log(2 * math.pi) + N * v * math.log(R)
    #print("Xi", Xi)
    l = k
    # Поиск критерия идентификации
    for l in range(N):
        if l == 0:
            x_tk = x_t0

        # Инициализация треугольничка
        Triangle = 0

        # Расчёт критерия индентификации для ложных тет
        x_tk_plus_one = np.matmul(F, x_tk) + np.dot(Psi, u_t0)
        Eps_tk_plus_one = y_true[l] - np.matmul(H, x_tk_plus_one)
        Triangle = Triangle + np.multiply(Eps_tk_plus_one.transpose(), Eps_tk_plus_one) * pow(R, -1)
        xi += Triangle


        x_tk = x_tk_plus_one
    return xi

def dXi(x):
    ###############___0.5____###################################
    # Zero point five


    #start_tets = [-2, 0.01]
    s = len(tetta_true)

    # 1
    F = np.array([(-0.8, 1), (x[0], 0)])
    PSI = np.array([x[1], 1])
    H = np.array([1, 0])

    dF = [np.array([(0, 0), (1, 0)]), np.array([(0, 0), (0, 0)])]
    dPSI = [[0, 0], [1, 0]]
    dH = [np.array([0, 0]), np.array([0, 0])]
    dR = (0, 0)

    gradient = [nu / 2 * N * 1 / R * 0 for alfa in range(s)]

    q_signals = 1
    u = [1 for _ in range(q_signals)]  # вектор управления (входа)
    signals_quantity = [1 for _ in range(q_signals)]  # сколько раз подали каждый входной сигнал (k_i)
    x = [np.array([0, 0]) for _ in range(len(u))]
    dx = x.copy()  # копируем, потому что дифференциал от нулей
    # цикл по N (момента времени?)
    l = k
    for l in range(N):
        delta = [0, 0]
        # цикл по сигналам
        for i in range(0, q_signals):

            x[i] = F.dot(x[i]) + PSI * u[i]
            # eps = y_true[l] - H.dot(x[i])
            # print(f"x[i] = {x[i]}, y = {y}")
            for alfa in range(s):
                dx[i] = dF[alfa].dot(x[i]) + F.dot(dx[i]) + dPSI[alfa] * u[i]
                deps = -dH[alfa].dot(x[i]) - H.dot(dx[i])
                # print(f"-{dH[alfa]}*{x[i]} - {H}*{dx[i]}")
                for j in range(signals_quantity[i]):
                    eps = y_true[l] - H.dot(x[i])
                    # eps = y - H.dot(x[i]) # по алгоритму вычисления эпсилон как будто должно быть здесь, но как по мне это неправильно
                    delta[alfa] += deps.transpose() / R * eps - eps.transpose() / (2 * R) * dR[alfa] / R * eps

                gradient[alfa] += delta[alfa]
    return(gradient)

if __name__ == '__main__':

    # Определение переменных
    k = 0
    m = q = i = j = v = nu = 1
    u_t0 = 1.0
    R = 0.1
    N = 20
    tetta_true = np.array([-1.5, 1.0])
    tetta_true = tetta_true.reshape(2, 1)

    tetta_else = np.array([-2.0, 0.1])
    tetta_else = tetta_else.reshape(2, 1)


    #Генерируем общий шум
    v_noise = np.dot(np.random.normal(0, 1.0, N), R)

    # Создаём вместилище для тетт
    tetta_all = np.zeros((5, 2))

    # Фиксируем правильный игрек для верных значений y
    y_true = y(tetta_true)

    # Отрисовка 3D графика
    # graf3D()

    # Третий пункт первой лабы
    ThirdPoint()





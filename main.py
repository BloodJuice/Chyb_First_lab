import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint

# return F, Psi, H, x_t0
def initVariables(tetta):
    F = np.array([[-0.8, 1.0], [tetta[0][0], 0]])
    Psi = np.array([[tetta[1][0]], [1.0]])
    H = np.array([[1, 0]])
    xt0 = np.array([[0], [0]])
    return F, Psi, H, xt0

def initGraduates(mode):
    if mode == 2:
        dF = [np.array([[0, 0], [1, 0]]), np.array([[0, 0], [0, 0]])]
        dPsi = [np.array([[0], [0]]), np.array([[1], [0]])]
        dH = [np.array([[0, 0]]), np.array([[0, 0]])]
        dR = [np.array([[0]]), np.array([[0]])]
        dx0 = [np.zeros((n, 1)) for i in range(s)]
        du_dua = 1
    elif mode == 1:
        dF = [np.array([[0]]), np.array([[0]])]
        dPsi = [np.array([[1, 0]]), np.array([[0, 1]])]
        dH = [np.array([[0]]), np.array([[0]])]
        dR = [np.array([[0]]), np.array([[0]])]
        dx0 = [np.zeros((n, 1)) for i in range(s)]
        du_dua = np.array([[[1], [0]], [[0], [1]]])
    return dF, dPsi, dH, dR, dx0, du_dua

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


def Minimize(tettaMin, yRun, plan, k, v, m, R):
    res = []
    ############__2__##########
    lim = [-2.0, 0.01, -0.05, 1.5]
    bounds = Bounds([lim[0], lim[1]],  # [min x0, min x1]
                    [lim[2], lim[3]])  # [max x0, max x1]
    result = minimize(fun=Xi, x0=tettaMin, args={"y": yRun,"plan": plan, "k": k, "v": v, "m": m, "R": R}, method='SLSQP', bounds=bounds)
    # res.append(minimize(Xi, x_start, method='SLSQP', jac=dXi, bounds=bounds))
    # print("Тетты для нулевого порядка:", result.__getitem__("x"))
    print("Тетты для первого порядка:", result)
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

def y(R, tetta, N, plan):
    F, Psi, H, x_t0 = initVariables(tetta)
    xtk = 0
    yEnd = []
    q = len(plan[0])
    print("\nplan size\n", np.shape(plan))
    for _ in range(q):
        yPlus = []
        for stepj in range(N):
            if stepj == 0:
                xtk = np.array([[0], [0]])
            xPlus = np.add(np.dot(F, xtk), np.dot(Psi, plan[_][stepj][0]))
            yPlus.append(np.add(np.dot(H, xPlus)[0][0], (np.random.normal(0, 1.) * R)))
            xtk = xPlus
        yEnd.append(np.array(yPlus).reshape(N, 1))
    yEnd = np.array(yEnd)
    return yEnd

def Xi(tetta, params):
    # Инициализация матриц/векторов, 1, 2 пункты:
    y = params['y'].copy()
    plan = params['plan'].copy()
    k = params['k'].copy()
    v = params['v']
    m = params['m']
    R = params['R']
    
    tetta = np.array(tetta).reshape(2, 1)
    F, Psi, H, xt0 = initVariables(tetta)
    q = len(k)
    xtk = np.array([[np.full(shape=2, fill_value=0, dtype=float).reshape(2, 1) for stepj in range(N)] for stepi in range(q)])
    xi = N * m * v * math.log(2 * math.pi) + N * v * math.log(R)  # Calculate Xi constant
    # Point 4
    u = plan

    for kCount in range(0, N):
        Triangle = 0  # Инициализация треугольничка
        for i in range(0, q):

            # Point 5
            if kCount == 0:
                xtk[i][kCount] = xt0

            # Расчёт критерия индентификации для ложных тет
            xPlusOne = np.dot(F, xtk[i][kCount]) + np.dot(Psi, u[i][kCount][0])

            for j in range(int(k[i])):
                epsPlusOne = y[i][kCount][0] - np.dot(H, xPlusOne)
                Triangle += np.multiply(epsPlusOne.transpose(), epsPlusOne) * pow(R, -1)
        xi += Triangle
        if kCount + 1 < N:
            xtk[i][kCount + 1] = xPlusOne
        else:
            xtk[i][kCount] = xPlusOne
    return 0.5 * xi

def dXi(x, params):
    ###############___0.5____###################################
    mode = 2
    s = len(tetta_true)
    y = params['y'].copy()
    plan = params['plan'].copy()
    k = params['k'].copy()
    v = params['v']
    m = params['m']
    R = params['R']
    N = params['N']

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

def KsiStart(q):
    ksi = []
    for stepi in range(q):
        Ui = []
        Ui = (np.full(shape=N, fill_value=1, dtype=float)).reshape(N, 1)
        ksi.append(Ui)
    return np.array(ksi)

def main(tettaTrue, tettaFalse):
    s, R, m = 2, 0.1, 1. # Nubmber of derivatives
    q = int(1 + s * (s + 1) / 2) # Number of k
    k = [1.0 for stepi in range(q)]  # Initial number of system start
    v = q



    startPlan = KsiStart(q)
    v_noise = (np.array(np.random.normal(0, 1., N) * R)).reshape(N, 1)
    yRun = y(R, tettaTrue, N, startPlan)
    Minimize(tettaFalse, yRun, startPlan, k, v, m, R)

    # print("\ntettaFalse:\n", Xi(tettaFalse, yRun, startPlan, k, v, N, m, R))
    # print("\ntettaTrue:\n", Xi(tettaTrue, yRun, startPlan, k, v, N, m, R))

if __name__ == '__main__':
    # Определение переменных
    r = 2  # Количество начальных сигналов, альфа
    n = 2  # Размерность вектора х0
    s = 2  # Количество производных по тетта
    N = 4  # Число испытаний

    tetta_true = (np.array([-1.5, 1.0])).reshape(2, 1)
    tetta_false = (np.array([-2.0, 0.1])).reshape(2, 1)

    main(tetta_true, tetta_false)







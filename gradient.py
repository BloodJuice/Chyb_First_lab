import math

import numpy as np

"""
Вопросы
1. Что такое Sp? След матрицы
2. Тета альфа это тета1-тета2 или вектор theta*? Это тета1-2
3. Что обозначают тета1 и тета2? Это диапазон изначальных значений, который мы берем для оптимизации
4. Как правильно дифференцировать матрицы по тета? Каждый элемент? Да, вроде как
5. Почему искомых параметра всего 2? Просто такова задача, могло бы быть и больше
6. Зависит ли вычисление градиента от вычисления самого критерия? Нет, задачи независимые

7. Что такое N? Количество сигналов? Тогда что такое q?
8. Что такое R?

Начальное значение градиента нулевое из-за производной по R
q - количество различных сигналов, подаваемых на вход
k_i - сколько раз был подан i-ый сигнал
j - реализация выходного сигнала, соответствующая i-му входному
"""


"""
theta1 = [-2; -0.05]
theta2 = [0.01; 1.5]
theta* = (-1.5, 1) 

k = [0; N-1], 
N = 50, 
q = 1, 
k_i = v = 1, 
i = j = 1, 
m = 1, 
u(tk) = 1,
s = 2
"""

N = 50

nu = 1
m = 1

true_tets = [-1.5, 1]
start_tets = [-2, 0.01]
s = len(true_tets)

# 1
F = np.array([(-0.8, 1), (start_tets[0], 0)])
PSI = np.array([start_tets[1], 1])
H = np.array([1, 0])

R = 0.1

# v_noise = R * np.random.normal(0, 1, N)
v_noise = R * np.random.normal(0, 1, N)

dF = [np.array([(0, 0), (1, 0)]), np.array([(0, 0), (0, 0)])]
dPSI = [[0, 0], [1, 0]]
dH = [np.array([0, 0]), np.array([0, 0])]
dR = (0, 0)

gradient = [nu/2 * N * 1/R * 0 for alfa in range(s)]

q_signals = 1
u = [1 for _ in range(q_signals)] # вектор управления (входа)
signals_quantity = [1 for _ in range(q_signals)] # сколько раз подали каждый входной сигнал (k_i)
x = [np.array([0, 0]) for _ in range(len(u))]
dx = x.copy() # копируем, потому что дифференциал от нулей
# цикл по N (момента времени?)
for k in range(N):
    delta = [0, 0]
    # цикл по сигналам
    for i in range(0, q_signals):

        x[i] = F.dot(x[i]) + PSI * u[i]
        y = H.dot(x[i]) + v_noise[k]
        eps = y - H.dot(x[i])
        # print(f"x[i] = {x[i]}, y = {y}")
        for alfa in range(s):
            dx[i] = dF[alfa].dot(x[i]) + F.dot(dx[i]) + dPSI[alfa] * u[i]
            deps = -dH[alfa].dot(x[i]) - H.dot(dx[i])
            # print(f"-{dH[alfa]}*{x[i]} - {H}*{dx[i]}")
            for j in range(signals_quantity[i]):
                # eps = y - H.dot(x[i]) # по алгоритму вычисления эпсилон как будто должно быть здесь, но как по мне это неправильно
                delta[alfa] += deps.transpose() / R * eps - eps.transpose() / 2 * R * \
                dR[alfa] / R  * eps

            gradient[alfa] += delta[alfa]

print(f"Градиент:\nЧастная производная по тета1 = {gradient[0]};\nЧастная производная по тета2 = {gradient[1]}")
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(10)
n = 30

def generate_data(n, eps = 0.1):
    x = np.linspace(-2, 2, n)
    r = np.random.rand(n) * eps
    y1 = x * (np.cos(r + 0.5 * x ** 3)) + np.sin(0.5 * x ** 3)
    y2 = 4*x**5 - 5*x**4 - 20*x**3 + 10*x**2 + 40*x + 10 + r

    return x, y1, y2


def create_design_matrix(x, n, m):
    A = np.ones((n, m))
    for i in range(1, m):
        A[:, i] = x**i

    return A

def plot_results(x, b1, b2):
    fig, ax = plt.subplots(2, 1)
    ax[0].scatter(x, b1, label='data set 1')
    ax[1].scatter(x, b2, label='data set 2')
    ax[0].legend()
    ax[1].legend()
    plt.show()


def forward_substitution(m, R, b):
    x = np.zeros(m)

    x[0] = b[0]/R[0,0]


    for i in range(1, m):
        temp = 0
        for j in range(0, i):
            temp += R[i,j]*x[j]

        x[i] = (b[i] - temp)/R[i,i]

    return x


def back_substitution(m, R, b):
    x = np.zeros(m)

    x[-1] = b[-1]/R[-1,-1]

    for i in range(m-2, -1, -1):
        temp = 0
        for j in range(i+1, m):
            temp += R[i, j]*x[j]
        x[i] = (b[i] - temp)/R[i, i]

    return x


def solve_qr(m, A, b):
    Q, R = np.linalg.qr(A)

    b1 = Q.T @ b

    x = back_substitution(m, R, b1)

    return A @ x


    fig, ax = plt.subplots(2, 1)
    ax[0].scatter(x, b1, label='data set 1')
    ax[1].scatter(x, b2, label='data set 2')
    ax[0].legend()
    ax[1].legend()
    plt.show()


def main():
    np.random.seed(10)
    n = 30
    m = 8

    x, y1, y2 = generate_data(n)
    A = create_design_matrix(x, n, m)


    y_tilde1 = solve_qr(m, A, y1)
    y_tilde2 = solve_qr(m, A, y2)
    print(y_tilde1)



    fig, ax = plt.subplots(2, 1)
    ax[0].scatter(x, y1, label='data set 1')
    ax[0].plot(x, y_tilde1, 'r--', label='least squares fit')
    ax[1].scatter(x, y2, label='data set 2')
    ax[1].plot(x, y_tilde2, 'r--', label='least squares fit')

    ax[0].legend()
    ax[1].legend()
    plt.show()



main()

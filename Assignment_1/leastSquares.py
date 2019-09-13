import numpy as np
import matplotlib.pyplot as plt



def generate_data(n, eps = 1):
    x = np.linspace(-2, 2, n)
    r = np.random.rand(n) * eps
    y1 = x * (np.cos(r + 0.5*x** 3) + np.sin(0.5*x**3))
    y2 = 4*x**5 - 5*x**4 - 20*x**3 + 10*x**2 + 40*x + 10 + r

    return x, y1, y2


def create_design_matrix(x, n, m):
    """
    Takes an array x and outputs the matrix
    1  x_1  x_1^2  ..  x_1^(m-1)
    1  x_2  x_2^2  ..  x_2^(m-1)
    .   .    .     ..   .
    .   .    .     ..   .
    1  x_n  x_n^2 ..  x_n^(m-1)
    """

    A = np.ones((n, m))
    for i in range(1, m):
        A[:, i] = x**i

    return A



def forward_substitution(m, R, b):
    """
    Solves the system Rx = b for a lower triangular matrix R
    """
    x = np.zeros(m)

    x[0] = b[0]/R[0,0]


    for i in range(1, m):
        temp = 0
        for j in range(0, i):
            temp += R[i,j]*x[j]

        x[i] = (b[i] - temp)/R[i,i]

    return x


def back_substitution(m, R, b):
    """
    Solves the system Rx = b for an upper triangular matrix R
    """
    x = np.zeros(m)

    x[-1] = b[-1]/R[-1,-1]

    for i in range(m-2, -1, -1):
        temp = 0
        for j in range(i+1, m):
            temp += R[i, j]*x[j]
        x[i] = (b[i] - temp)/R[i, i]

    return x


def solve_cholesky(m, A, b):
    """
    Solves the system Ax = b by computing the Cholesky factorization of A = RR^T
    and returns the product Ax
    """
    AT = A.T
    B = AT @ A

    L = np.zeros((m,m))
    D = np.zeros((m,m))


    for k in range(m):
        L[:,k] = B[:,k]/B[k,k]
        D[k,k] = B[k,k]
        B = B - D[k,k]*np.outer(L[:,k], L[:,k].T)


    R = L @ np.sqrt(D)

    w = forward_substitution(m, R, AT @ b)
    x = back_substitution(m, R.T, w)


    return A @ x




def solve_qr(m, A, b):
    """
    Uses the factorization of A = QR and performs back substitution
    to compute the product of Ax
    """
    Q, R = np.linalg.qr(A)
    b1 = Q.T @ b

    x = back_substitution(m, R, b1)

    # print(x)

    return A @ x




def main():
    np.random.seed(0)
    n = 30
    m = 8

    x, y1, y2 = generate_data(n, eps=1)
    A = create_design_matrix(x, n, m)

    solve_cholesky(m ,A, y1)

    y_qr1 = solve_qr(m, A, y1)
    y_qr2 = solve_qr(m, A, y2)

    y_tilde1 = solve_cholesky(m, A, y1)
    y_tilde2 = solve_cholesky(m, A, y2)


    fig, ax = plt.subplots(2, 1)
    ax[0].scatter(x, y1, label='data set 1')
    ax[0].plot(x, y_tilde1, 'r--', label='least squares fit')
    ax[1].scatter(x, y2, label='data set 2')
    ax[1].plot(x, y_tilde2, 'r--', label='least squares fit')

    ax[0].legend()
    ax[1].legend()
    plt.show()



main()

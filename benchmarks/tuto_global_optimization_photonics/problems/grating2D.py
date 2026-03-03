import numpy as np
from scipy.linalg import toeplitz
from .photonic_problem import photonic_problem


class grating2D(photonic_problem):
    def __init__(self, nb_layers, min_w, max_w, min_thick, max_thick, min_p, max_p):
        super().__init__()
        self.nb_layers = nb_layers
        self.n = 3 * nb_layers
        self.min_w = min_w
        self.max_w = max_w
        self.min_thick = min_thick
        self.max_thick = max_thick
        self.min_p = min_p
        self.max_p = max_p
        self.lb = np.array(
            [min_w] * nb_layers + [min_thick] * nb_layers + [min_p] * nb_layers
        )
        self.ub = np.array(
            [max_w] * nb_layers + [max_thick] * nb_layers + [max_p] * nb_layers
        )

    # i = complex(0, 1)

    # RCWA functions
    def cascade(self, T, U):
        n = int(T.shape[1] / 2)
        J = np.linalg.inv(np.eye(n) - np.matmul(U[0:n, 0:n], T[n : 2 * n, n : 2 * n]))
        K = np.linalg.inv(np.eye(n) - np.matmul(T[n : 2 * n, n : 2 * n], U[0:n, 0:n]))
        S = np.block(
            [
                [
                    T[0:n, 0:n]
                    + np.matmul(
                        np.matmul(np.matmul(T[0:n, n : 2 * n], J), U[0:n, 0:n]),
                        T[n : 2 * n, 0:n],
                    ),
                    np.matmul(np.matmul(T[0:n, n : 2 * n], J), U[0:n, n : 2 * n]),
                ],
                [
                    np.matmul(np.matmul(U[n : 2 * n, 0:n], K), T[n : 2 * n, 0:n]),
                    U[n : 2 * n, n : 2 * n]
                    + np.matmul(
                        np.matmul(
                            np.matmul(U[n : 2 * n, 0:n], K), T[n : 2 * n, n : 2 * n]
                        ),
                        U[0:n, n : 2 * n],
                    ),
                ],
            ]
        )
        return S

    def c_bas(self, A, V, h):
        n = int(A.shape[1] / 2)
        D = np.diag(np.exp(1j * V * h))
        S = np.block(
            [
                [A[0:n, 0:n], np.matmul(A[0:n, n : 2 * n], D)],
                [
                    np.matmul(D, A[n : 2 * n, 0:n]),
                    np.matmul(np.matmul(D, A[n : 2 * n, n : 2 * n]), D),
                ],
            ]
        )
        return S

    def marche(self, a, b, p, n, x):
        l = np.zeros(n, dtype=np.complex128)
        m = np.zeros(n, dtype=np.complex128)
        tmp = (
            1
            / (2 * np.pi * np.arange(1, n))
            * (np.exp(-2 * 1j * np.pi * p * np.arange(1, n)) - 1)
            * np.exp(-2 * 1j * np.pi * np.arange(1, n) * x)
        )
        l[1:n] = 1j * (a - b) * tmp
        l[0] = p * a + (1 - p) * b
        m[0] = l[0]
        m[1:n] = 1j * (b - a) * np.conj(tmp)
        T = toeplitz(l, m)
        return T

    def creneau(self, k0, a0, pol, e1, e2, a, n, x0):
        nmod = int(n / 2)
        alpha = np.diag(a0 + 2 * np.pi * np.arange(-nmod, nmod + 1))
        if pol == 0:
            M = alpha * alpha - k0 * k0 * self.marche(e1, e2, a, n, x0)
            L, E = np.linalg.eig(M)
            L = np.sqrt(-L + 0j)
            L = (1 - 2 * (np.imag(L) < -1e-15)) * L
            P = np.block([[E], [np.matmul(E, np.diag(L))]])
        else:
            U = self.marche(1 / e1, 1 / e2, a, n, x0)
            T = np.linalg.inv(U)
            M = (
                np.matmul(
                    np.matmul(
                        np.matmul(T, alpha),
                        np.linalg.inv(self.marche(e1, e2, a, n, x0)),
                    ),
                    alpha,
                )
                - k0 * k0 * T
            )
            L, E = np.linalg.eig(M)
            L = np.sqrt(-L + 0j)
            L = (1 - 2 * (np.imag(L) < -1e-15)) * L
            P = np.block([[E], [np.matmul(np.matmul(U, E), np.diag(L))]])
        return P, L

    def homogene(self, k0, a0, pol, epsilon, n):
        nmod = int(n / 2)
        valp = np.sqrt(
            epsilon * k0 * k0 - (a0 + 2 * np.pi * np.arange(-nmod, nmod + 1)) ** 2 + 0j
        )
        valp = valp * (1 - 2 * (valp < 0)) * (pol / epsilon + (1 - pol))
        P = np.block([[np.eye(n)], [np.diag(valp)]])
        return P, valp

    def interface(self, P, Q):
        n = int(P.shape[1])
        S = np.matmul(
            np.linalg.inv(
                np.block(
                    [
                        [P[0:n, 0:n], -Q[0:n, 0:n]],
                        [P[n : 2 * n, 0:n], Q[n : 2 * n, 0:n]],
                    ]
                )
            ),
            np.block(
                [[-P[0:n, 0:n], Q[0:n, 0:n]], [P[n : 2 * n, 0:n], Q[n : 2 * n, 0:n]]]
            ),
        )
        return S

    # Cost function
    def __call__(self, x):
        lam_blue = 449.5897
        pol = 1
        d = 600.521475
        nmod = 25
        e2 = 2.4336
        n = 2 * nmod + 1
        n_motifs = int(x.size / 3)
        x = x / d
        h = x[n_motifs : 2 * n_motifs]
        x0 = x[2 * n_motifs : 3 * n_motifs]
        a = x[0:n_motifs]
        spacers = np.zeros(a.size)

        # Maximization of the blue specular reflection
        l = lam_blue / d
        k0 = 2 * np.pi / l
        P, V = self.homogene(k0, 0, pol, 1, n)
        S = np.block(
            [
                [np.zeros([n, n]), np.eye(n, dtype=np.complex128)],
                [np.eye(n), np.zeros([n, n])],
            ]
        )
        for j in range(0, n_motifs):
            Pc, Vc = self.creneau(k0, 0, pol, e2, 1, a[j], n, x0[j])
            S = self.cascade(S, self.interface(P, Pc))
            S = self.c_bas(S, Vc, h[j])
            S = self.cascade(S, self.interface(Pc, P))
            S = self.c_bas(S, V, spacers[j])
        Pc, Vc = self.homogene(k0, 0, pol, e2, n)
        S = self.cascade(S, self.interface(P, Pc))
        R = np.zeros(3, dtype=float)
        for j in range(-1, 2):
            R[j] = abs(S[j + nmod, nmod]) ** 2 * np.real(V[j + nmod]) / k0
        cost = 1 - (R[-1] + R[1]) / 2 + R[0] / 2

        return cost

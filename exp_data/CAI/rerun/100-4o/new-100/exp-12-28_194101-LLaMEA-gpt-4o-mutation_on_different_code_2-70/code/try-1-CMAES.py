import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.sigma = 0.5
        self.lambda_ = 4 + int(3 * np.log(self.dim))
        self.mu = self.lambda_ // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights ** 2)
        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.c_sigma
        self.cc = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.dim + 2) ** 2 + self.mu_eff))
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.C = self.B @ np.diag(self.D**2) @ self.B.T
        self.chi_n = self.dim**0.5 * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))
        self.x_mean = np.random.normal(0, 1, self.dim)  # Change: Initialize x_mean with Gaussian distribution
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        for _ in range(0, self.budget, self.lambda_):
            arz = np.random.randn(self.lambda_, self.dim)
            y = self.B @ (self.D[:, np.newaxis] * arz.T).T
            x = self.x_mean + self.sigma * y
            x = np.clip(x, func.bounds.lb, func.bounds.ub)
            fitness = np.array([func(ind) for ind in x])
            idx = np.argsort(fitness)
            x = x[idx]
            y = y[idx]
            self.x_mean = np.sum(self.weights[:, np.newaxis] * x[:self.mu], axis=0)
            y_w = np.sum(self.weights[:, np.newaxis] * y[:self.mu], axis=0)
            self.ps = (1 - self.c_sigma) * self.ps + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * (self.B @ y_w / self.D)
            hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.c_sigma) ** (2 * (_ / self.lambda_ + 1))) / self.chi_n < 1.4 + 2 / (self.dim + 1)
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y_w
            self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * (self.B @ np.diag(1 / self.D) @ y[:self.mu].T) @ np.diag(self.weights) @ (self.B @ np.diag(1 / self.D) @ y[:self.mu].T).T
            if np.any(np.isinf(fitness)) or np.any(np.isnan(fitness)):
                continue
            if fitness[0] < self.f_opt:
                self.f_opt = fitness[0]
                self.x_opt = x[0]
            self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (np.linalg.norm(self.ps) / self.chi_n - 1))
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(self.D)
        return self.f_opt, self.x_opt
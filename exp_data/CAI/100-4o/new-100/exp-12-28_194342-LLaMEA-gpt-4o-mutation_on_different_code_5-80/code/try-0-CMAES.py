import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10, sigma=0.3):
        self.budget = budget
        self.dim = dim
        self.sigma = sigma
        self.population_size = 4 + int(3 * np.log(dim))
        self.lambd = self.population_size
        self.weights = np.log(self.population_size + 0.5) - np.log(np.arange(1, self.population_size + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights**2)
        self.c_c = (4 + self.mu_eff / self.dim) / (dim + 4 + 2 * self.mu_eff / self.dim)
        self.c_s = (self.mu_eff + 2) / (dim + self.mu_eff + 3)
        self.c_1 = 2 / ((dim + 1.3)**2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((dim + 2)**2 + self.mu_eff))
        self.damping = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (dim + 1)) - 1) + self.c_s
        self.p_c = np.zeros(dim)
        self.p_s = np.zeros(dim)
        self.B = np.eye(dim)
        self.D = np.ones(dim)
        self.C = np.eye(dim)
        self.chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        mean = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        evals = 0
        while evals < self.budget:
            z = np.random.randn(self.lambd, self.dim)
            y = z @ np.diag(self.D) @ self.B.T
            x = mean + self.sigma * y
            fitness = np.array([func(x[i]) for i in range(self.lambd)])
            evals += self.lambd
            indices = np.argsort(fitness)
            x = x[indices]
            z = z[indices]
            mean = np.dot(self.weights, x[:self.lambd])
            y_w = np.dot(self.weights, z[:self.lambd])
            self.p_s = (1 - self.c_s) * self.p_s + np.sqrt(self.c_s * (2 - self.c_s) * self.mu_eff) * y_w
            norm_p_s = np.linalg.norm(self.p_s)
            h_sig = int(norm_p_s / np.sqrt(1 - (1 - self.c_s)**(2 * evals / self.lambd)) < (1.4 + 2 / (self.dim + 1)) * self.chi_n)
            self.p_c = (1 - self.c_c) * self.p_c + h_sig * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * np.dot(self.weights, y[:self.lambd])
            self.C = (1 - self.c_1 - self.c_mu) * self.C + self.c_1 * np.outer(self.p_c, self.p_c) + self.c_mu * np.dot((np.dot(y[:self.lambd].T, np.diag(self.weights))), y[:self.lambd])
            self.sigma *= np.exp((self.c_s / self.damping) * (norm_p_s / self.chi_n - 1))
            self.B, self.D = np.linalg.eigh(self.C)
            self.D = np.sqrt(np.maximum(self.D, 0))
            if fitness[0] < self.f_opt:
                self.f_opt = fitness[0]
                self.x_opt = x[0]
        return self.f_opt, self.x_opt
import numpy as np

class CMA_ES:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.sigma = 0.3  # Step size
        self.pop_size = 4 + int(3 * np.log(dim))  # Population size
        self.weights = np.log(self.pop_size / 2 + 0.5) - np.log(np.arange(1, self.pop_size + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights**2)
        self.c_sigma = (self.mu_eff + 2) / (dim + self.mu_eff + 5)
        self.c_c = 4 / (dim + 4)
        self.c_1 = 2 / ((dim + 1.3)**2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((dim + 2)**2 + self.mu_eff))
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (dim + 1)) - 1) + self.c_sigma
        self.chi_n = np.sqrt(dim) * (1 - 1 / (4*dim) + 1 / (21*dim**2))

    def __call__(self, func):
        mean = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        cov = np.eye(self.dim)
        ps = np.zeros(self.dim)
        pc = np.zeros(self.dim)
        func_evals = 0

        while func_evals < self.budget:
            arz = np.random.randn(self.pop_size, self.dim)
            y = mean + self.sigma * arz.dot(np.linalg.cholesky(cov).T)
            f_values = np.array([func(ind) for ind in y])
            func_evals += self.pop_size

            sort_indices = np.argsort(f_values)
            y_sorted = y[sort_indices]
            best_parents = y_sorted[:int(self.pop_size / 2)]
            mean = np.dot(self.weights, best_parents)

            ps = (1 - self.c_sigma) * ps + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * np.dot(np.linalg.inv(np.linalg.cholesky(cov).T), (mean - y_sorted[0]) / self.sigma)
            pc = (1 - self.c_c) * pc + np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * (mean - y_sorted[0]) / self.sigma

            C_new = (1 - self.c_1 - self.c_mu) * cov + self.c_1 * np.outer(pc, pc)
            for k in range(int(self.pop_size / 2)):
                y_k = (y_sorted[k] - y_sorted[0]) / self.sigma
                C_new += self.c_mu * self.weights[k] * np.outer(y_k, y_k)
            cov = C_new

            self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (np.linalg.norm(ps) / self.chi_n - 1))

            if f_values[sort_indices[0]] < self.f_opt:
                self.f_opt = f_values[sort_indices[0]]
                self.x_opt = y_sorted[0]

        return self.f_opt, self.x_opt
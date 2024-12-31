import numpy as np

class CMAESElitism:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.lambda_ = 4 + int(3 * np.log(self.dim))  # offspring size
        self.mu = self.lambda_ // 2  # number of parents
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights ** 2)
        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.c_c = 4 / (self.dim + 4)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mu_eff)
        self.c_mu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.dim + 2)**2 + self.mu_eff))
        self.d_sigma = 1 + self.c_sigma + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1)
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        x_mean = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        sigma = (func.bounds.ub - func.bounds.lb) * 0.3
        ps = np.zeros(self.dim)
        pc = np.zeros(self.dim)
        C = np.eye(self.dim)
        eigen_eval = 0
        count_eval = 0

        while count_eval < self.budget:
            if count_eval - eigen_eval > self.lambda_ / (self.c1 + self.c_mu) / self.dim / 10:
                C = np.triu(C) + np.triu(C, 1).T
                D, B = np.linalg.eigh(C)
                eigen_eval = count_eval

            D = np.maximum(D, 1e-10)
            B_D = B * np.sqrt(D)

            population = np.array([x_mean + sigma * np.dot(B_D, np.random.randn(self.dim)) for _ in range(self.lambda_)])
            fitness = [func(x) for x in population]
            count_eval += self.lambda_

            indices = np.argsort(fitness)
            if fitness[indices[0]] < self.f_opt:
                self.f_opt = fitness[indices[0]]
                self.x_opt = population[indices[0]]

            x_mean_old = x_mean
            x_mean = np.dot(self.weights, population[indices[:self.mu]])
            y = x_mean - x_mean_old
            z = np.dot(B.T, y) / sigma

            ps = (1 - self.c_sigma) * ps + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * z
            h_sigma = np.linalg.norm(ps) / np.sqrt(1 - (1 - self.c_sigma) ** (2 * count_eval / self.lambda_)) < (1.4 + 2 / (self.dim + 1)) * (1 - 1 / (4 * self.dim + 1))
            pc = (1 - self.c_c) * pc + h_sigma * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y

            C = ((1 - self.c1 - self.c_mu) * C
                 + self.c1 * np.outer(pc, pc)
                 + self.c_mu * np.dot((np.array([self.weights[i] * np.outer(population[indices[i]] - x_mean_old, population[indices[i]] - x_mean_old) / sigma ** 2 for i in range(self.mu)]).T), np.eye(self.dim)))

            sigma *= np.exp((np.linalg.norm(ps) / np.sqrt(1 - (1 - self.c_sigma) ** (2 * count_eval / self.lambda_)) - 1) * self.c_sigma / self.d_sigma)

        return self.f_opt, self.x_opt
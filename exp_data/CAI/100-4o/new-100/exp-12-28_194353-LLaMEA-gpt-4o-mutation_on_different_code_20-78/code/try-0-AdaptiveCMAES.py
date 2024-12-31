import numpy as np

class AdaptiveCMAES:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.sigma = 0.3  # Initial step-size
        self.population_size = 4 + int(3 * np.log(self.dim))
        self.covariance_matrix = np.eye(self.dim)
        self.mean = np.random.uniform(-5.0, 5.0, self.dim)
        self.weights = np.log(self.population_size + 0.5) - np.log(np.arange(1, self.population_size + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights**2)
        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 3)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.c_sigma
        self.path_c = np.zeros(self.dim)
        self.path_sigma = np.zeros(self.dim)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        for _ in range(self.budget // self.population_size):
            z = np.random.randn(self.population_size, self.dim)
            y = np.dot(z, np.linalg.cholesky(self.covariance_matrix).T)
            x = self.mean + self.sigma * y
            x = np.clip(x, lb, ub)
            f_values = np.array([func(ind) for ind in x])

            sorted_indices = np.argsort(f_values)
            x = x[sorted_indices]
            z = z[sorted_indices]

            if f_values[0] < self.f_opt:
                self.f_opt = f_values[0]
                self.x_opt = x[0]
                
            y_mean = np.dot(self.weights, z)
            self.mean = np.dot(self.weights, x)

            self.path_sigma = (1 - self.c_sigma) * self.path_sigma + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * y_mean
            self.sigma *= np.exp((np.linalg.norm(self.path_sigma) / np.sqrt(self.dim)) - 1) / self.d_sigma

            h_sigma = int((np.linalg.norm(self.path_sigma) / np.sqrt(1 - (1 - self.c_sigma) ** (2 * (_ + 1))) / (1.4 + 2 / (self.dim + 1))) < (1.4 + 2 / (self.dim + 1)))
            self.path_c = (1 - self.c_sigma) * self.path_c + h_sigma * np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * y_mean
            c1 = 2 / ((self.dim + 1.3)**2 + self.mu_eff)
            cmu = min(1 - c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.dim + 2)**2 + self.mu_eff))
            self.covariance_matrix *= (1 - c1 - cmu * np.sum(self.weights))
            self.covariance_matrix += c1 * np.outer(self.path_c, self.path_c)
            self.covariance_matrix += cmu * np.dot((self.weights[:, np.newaxis] * z).T, z)

        return self.f_opt, self.x_opt
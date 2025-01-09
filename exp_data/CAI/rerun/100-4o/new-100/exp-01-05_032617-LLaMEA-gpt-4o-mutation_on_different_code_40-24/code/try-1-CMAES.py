import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.mean = np.random.uniform(-5, 5, dim)
        self.sigma = 0.3
        self.lambda_ = 4 + int(3 * np.log(dim))
        self.weights = np.log(self.lambda_ / 2 + 0.5) - np.log(np.arange(1, self.lambda_ + 1))
        self.weights /= self.weights.sum()
        self.mu_eff = 1 / np.sum(self.weights ** 2)
        self.cov_matrix = np.eye(dim)
        self.evo_path = np.zeros(dim)
        self.cov_path = np.zeros(dim)
        self.damping = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (dim + 1)) - 1) + 0.3
        self.ccov = 2 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.cc = 2 / (dim + 2)
        self.c_sigma = (self.mu_eff + 2) / (dim + self.mu_eff + 3)

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            samples = np.random.multivariate_normal(self.mean, self.sigma**2 * self.cov_matrix, self.lambda_)
            mirrored_samples = 2 * self.mean - samples
            all_samples = np.vstack((samples, mirrored_samples))
            fitness = np.array([func(x) for x in all_samples])
            evals += len(all_samples)
            
            indices = np.argsort(fitness)
            all_samples = all_samples[indices]
            fitness = fitness[indices]
            self.mean = np.dot(self.weights, all_samples[:self.mu_eff])

            y = self.mean - np.mean(all_samples[:self.mu_eff], axis=0)
            norm_y = np.linalg.norm(np.dot(np.linalg.inv(self.cov_matrix), y))
            self.evo_path = (1 - self.cc) * self.evo_path + np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y / self.sigma

            hsig = (np.linalg.norm(self.evo_path) / np.sqrt(1 - (1 - self.cc)**(2 * evals/self.lambda_))) < (1.4 + 2 / (dim + 1))
            adaptive_cov_path = (1 - self.ccov) * self.cov_path + hsig * np.sqrt(self.ccov * (2 - self.ccov) * self.mu_eff) * y
            self.cov_matrix = (1 - self.ccov) * self.cov_matrix + self.ccov * np.outer(adaptive_cov_path, adaptive_cov_path)

            self.sigma *= np.exp((np.linalg.norm(self.evo_path) / np.sqrt(1 - (1 - self.cc)**(2 * evals/self.lambda_)) - 1) * self.c_sigma)

            if fitness[0] < self.f_opt:
                self.f_opt = fitness[0]
                self.x_opt = all_samples[0]

        return self.f_opt, self.x_opt
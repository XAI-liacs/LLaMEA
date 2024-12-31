import numpy as np

class AdaptiveCMAES:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 4 + int(3 * np.log(self.dim))
        self.sigma = 0.3
        self.covariance = np.eye(self.dim)
        self.mean = np.random.uniform(-5.0, 5.0, self.dim)
        self.evolution_path = np.zeros(self.dim)
        self.adaptive_inc = 1.0

    def __call__(self, func):
        lambda_ = self.population_size
        weights = np.log(lambda_ / 2 + 0.5) - np.log(np.arange(1, lambda_ + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_cov = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 3)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma

        for _ in range(int(self.budget / lambda_)):
            samples = np.random.multivariate_normal(self.mean, self.sigma**2 * self.covariance, lambda_)
            samples = np.clip(samples, -5.0, 5.0)
            fitness = np.array([func(x) for x in samples])
            
            indices = np.argsort(fitness)
            samples = samples[indices]
            fitness = fitness[indices]

            if fitness[0] < self.f_opt:
                self.f_opt = fitness[0]
                self.x_opt = samples[0]
                c_cov *= 0.9  # Adaptive covariance scaling when finding better solution

            z_mean = np.dot(weights, samples - self.mean) / self.sigma
            self.mean += self.sigma * z_mean

            self.evolution_path = (1 - c_c) * self.evolution_path + np.sqrt(c_c * (2 - c_c) * mu_eff) * z_mean
            h_sigma = int(np.linalg.norm(self.evolution_path) / np.sqrt(1 - (1 - c_sigma)**(2 * (_ + 1))) < 1.4 + 2 / (self.dim + 1))
            
            self.covariance = (1 - c_cov) * self.covariance + c_cov * np.outer(self.evolution_path, self.evolution_path) * h_sigma
            self.sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(self.evolution_path) / np.sqrt(1 - (1 - c_sigma)**(2 * (_ + 1))) - 1))

        return self.f_opt, self.x_opt
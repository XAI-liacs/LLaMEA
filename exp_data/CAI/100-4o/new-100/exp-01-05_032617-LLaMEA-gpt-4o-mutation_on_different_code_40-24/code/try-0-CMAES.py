import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.mean = np.random.uniform(-5, 5, dim)
        self.sigma = 0.3 # Initial step size
        self.lambda_ = 4 + int(3 * np.log(dim)) # Population size
        self.weights = np.log(self.lambda_ / 2 + 0.5) - np.log(np.arange(1, self.lambda_ + 1))
        self.weights /= self.weights.sum() # Normalize weights
        self.mu = int(self.lambda_ / 2)
        self.cov_matrix = np.eye(dim)
        self.evo_path = np.zeros(dim)
        self.cov_path = np.zeros(dim)
        self.damping = 1 + 2 * max(0, np.sqrt((self.mu - 1) / (dim + 1)) - 1) + 0.3
        self.ccov = 4 / (dim + 4)
        self.cc = (4 + self.mu / dim) / (dim + 4 + 2 * self.mu / dim)

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            # Sample new population
            samples = np.random.multivariate_normal(self.mean, self.sigma**2 * self.cov_matrix, self.lambda_)
            fitness = np.array([func(x) for x in samples])
            evals += self.lambda_
            
            # Sort by fitness and update the mean
            indices = np.argsort(fitness)
            samples = samples[indices]
            fitness = fitness[indices]
            self.mean = np.dot(self.weights, samples[:self.mu])

            # Update paths
            y = self.mean - np.mean(samples[:self.mu], axis=0)
            norm_y = np.linalg.norm(np.dot(np.linalg.inv(self.cov_matrix), y))
            self.evo_path = (1 - self.cc) * self.evo_path + np.sqrt(self.cc * (2 - self.cc) * self.mu) * y / self.sigma
            self.cov_path = (1 - self.ccov) * self.cov_path + np.sqrt(self.ccov * (2 - self.ccov) * self.mu) * y

            # Adapt covariance matrix
            self.cov_matrix = (1 - self.ccov) * self.cov_matrix + self.ccov * np.outer(self.cov_path, self.cov_path)
            
            # Adapt step size
            self.sigma *= np.exp((np.linalg.norm(self.evo_path) / np.sqrt(1 - (1 - self.cc)**(2 * evals/self.lambda_)) - 1) * self.damping)

            # Update best solution found
            if fitness[0] < self.f_opt:
                self.f_opt = fitness[0]
                self.x_opt = samples[0]

        return self.f_opt, self.x_opt
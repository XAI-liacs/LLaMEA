import numpy as np

class CMA_ES:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.sigma = 0.3  # Step size
        self.population_size = 4 + int(3 * np.log(dim))
        self.weights = np.log(self.population_size / 2 + 0.5) - np.log(np.arange(1, self.population_size + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights**2)
        self.c1 = 2 / ((dim + 1.3)**2 + self.mu_eff)
        self.c_mu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((dim + 2)**2 + self.mu_eff))
        self.cc = 4 / (dim + 4)
        self.cs = (self.mu_eff + 2) / (dim + self.mu_eff + 5)
        self.damps = 1 + self.cs + 2 * max(0, np.sqrt((self.mu_eff - 1) / (dim + 1)) - 1)
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.C = np.eye(dim)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        mean = np.random.uniform(lb, ub, self.dim)
        
        for _ in range(self.budget // self.population_size):
            # Sample population
            chol_cov = np.linalg.cholesky(self.C)
            samples = np.random.randn(self.population_size, self.dim)
            population = mean + self.sigma * (samples @ chol_cov.T)
            population = np.clip(population, lb, ub)

            # Evaluate and sort population
            fitness = np.array([func(ind) for ind in population])
            order = np.argsort(fitness)
            population = population[order]
            fitness = fitness[order]

            # Update best solution
            if fitness[0] < self.f_opt:
                self.f_opt = fitness[0]
                self.x_opt = population[0]

            # Recombination
            mean_old = mean
            mean = np.sum(self.weights[:, np.newaxis] * population[:len(self.weights)], axis=0)

            # Update evolution paths
            y = mean - mean_old
            z = samples[:len(self.weights)] @ self.weights
            norm_z = np.linalg.norm(z)
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * z

            hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * (self.budget // self.population_size))) / 1.4 < 1.4 + 2 / (self.dim + 1)
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y

            # Covariance matrix adaptation
            self.C = (1 - self.c1 - self.c_mu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) \
                     + self.c_mu * np.sum([w * np.outer(p, p) for w, p in zip(self.weights[:len(self.weights)], samples[:len(self.weights)])], axis=0)

            # Update step-size
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / norm_z - 1))

        return self.f_opt, self.x_opt
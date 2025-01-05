import numpy as np

class ImprovedAdaptiveCMAES:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5.0, 5.0)
        self.population_size = 4 + int(3 * np.log(self.dim))
        self.sigma = 0.3 * (self.bounds[1] - self.bounds[0])
        self.m = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.weights = np.log(self.population_size + 0.5) - np.log(np.arange(1, self.population_size + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1 / np.sum(self.weights**2)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((self.dim + 2)**2 + self.mueff))
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 3)
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        self.cc = 4 / (self.dim + 4)
        self.chiN = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        self.restart_threshold = self.sigma * 1e-3

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            # Mirrored sampling
            half_pop = self.population_size // 2
            population = np.random.multivariate_normal(self.m, self.sigma**2 * self.C, half_pop)
            population = np.vstack([population, 2*self.m - population])
            population = np.clip(population, self.bounds[0], self.bounds[1])
            
            fitness = np.array([func(ind) for ind in population])
            evals += len(fitness)
            
            indices = np.argsort(fitness)
            population = population[indices]
            fitness = fitness[indices]
            
            if fitness[0] < self.f_opt:
                self.f_opt = fitness[0]
                self.x_opt = population[0]
            
            self.m = np.dot(self.weights, population[:len(self.weights)])
            y_w = np.dot(self.weights, population[:len(self.weights)] - self.m) / self.sigma
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * np.dot(self.C, y_w)
            hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * evals / self.population_size)) / self.chiN < 1.4 + 2/(self.dim + 1)
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w
            artmp = (population[:len(self.weights)] - self.m) / self.sigma
            self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * np.dot((self.weights * artmp).T, artmp)
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
            
            if self.sigma < self.restart_threshold:
                self.sigma = 0.3 * (self.bounds[1] - self.bounds[0])
                self.m = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
                self.C = np.eye(self.dim)

        return self.f_opt, self.x_opt
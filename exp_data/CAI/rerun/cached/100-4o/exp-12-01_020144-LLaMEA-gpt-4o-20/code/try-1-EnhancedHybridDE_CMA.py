import numpy as np

class EnhancedHybridDE_CMA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(4, int(self.budget / (10 * dim)))
        self.population = np.random.uniform(-5, 5, (self.pop_size, dim))
        self.sigma = 0.3
        self.weights = np.log(self.pop_size / 2 + 0.5) - np.log(np.arange(1, self.pop_size + 1))
        self.weights /= self.weights.sum()
        self.mu_eff = 1 / np.sum(self.weights**2)
        self.cov_matrix = np.eye(dim)
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.cs = (self.mu_eff + 2) / (dim + self.mu_eff + 3)
        self.convergence_criterion = 10 - min(1e-6, 1 / dim)
    
    def __call__(self, func):
        count_evals = 0
        population_values = np.array([func(ind) for ind in self.population])
        count_evals += self.pop_size
        
        while count_evals < self.budget:
            indices = np.random.permutation(self.pop_size)
            for i in range(self.pop_size):
                if count_evals >= self.budget:
                    break
                x1, x2, x3 = self.population[indices[i]], self.population[indices[(i+1) % self.pop_size]], self.population[indices[(i+2) % self.pop_size]]
                scale_factor = np.random.uniform(0.5, 1.0)  # Adaptive mutation scaling
                mutant = x1 + scale_factor * (x2 - x3)
                mutant = np.clip(mutant, -5, 5)
                cross_points = np.random.rand(self.dim) < 0.9
                trial = np.where(cross_points, mutant, self.population[i])
                trial_value = func(trial)
                count_evals += 1
                if trial_value < population_values[i]:
                    self.population[i] = trial
                    population_values[i] = trial_value
            
            best_indices = np.argsort(population_values)
            self.population = self.population[best_indices[:self.mu_eff]]  # Select top mu_eff individuals
            population_values = population_values[best_indices[:self.mu_eff]]
            mean = np.dot(self.weights, self.population)
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * np.dot(np.linalg.inv(self.cov_matrix), mean - self.population[0])
            self.pc = (1 - self.cs) * self.pc + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (mean - self.population[0])
            self.cov_matrix = (1 - self.cs) * self.cov_matrix + self.cs * np.outer(self.pc, self.pc)
            
            if np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * count_evals / self.pop_size)) < self.convergence_criterion:
                self.sigma *= np.exp(0.2 + self.cs * (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * count_evals / self.pop_size)) - 1))
            
            for i in range(self.pop_size):
                if count_evals >= self.budget:
                    break
                self.population[i] = np.random.multivariate_normal(mean, self.sigma**2 * self.cov_matrix)
                np.clip(self.population[i], -5, 5, out=self.population[i])
                population_values[i] = func(self.population[i])
                count_evals += 1
        return self.population[np.argmin(population_values)]
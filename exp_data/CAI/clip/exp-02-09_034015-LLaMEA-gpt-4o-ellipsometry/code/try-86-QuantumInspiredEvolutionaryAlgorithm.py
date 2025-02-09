import numpy as np

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.1  # Quantum rotation angle
        self.beta = 0.9   # Collapse probability factor
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        theta = np.random.uniform(0, np.pi, (self.population_size, self.dim))
        population = np.zeros((self.population_size, self.dim))
        
        for i in range(self.population_size):
            population[i] = lb + (ub - lb) * (np.sin(theta[i]) ** 2)
        
        fitness = np.apply_along_axis(func, 1, population)
        eval_count = self.population_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        while eval_count < self.budget:
            new_population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                theta[i] += self.alpha * (2 * np.pi * np.random.rand(self.dim) - np.pi)
                new_population[i] = lb + (ub - lb) * (np.sin(theta[i]) ** 2)

            if np.random.rand() < self.beta:
                collapse_idx = np.random.randint(self.population_size)
                theta[collapse_idx] = np.arctan2(np.sqrt(np.random.rand(self.dim)), np.ones(self.dim))
                new_population[collapse_idx] = lb + (ub - lb) * (np.sin(theta[collapse_idx]) ** 2)

            new_population = np.clip(new_population, lb, ub)
            new_fitness = np.apply_along_axis(func, 1, new_population)
            eval_count += self.population_size

            for i in range(self.population_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    if new_fitness[i] < best_fitness:
                        best_solution = new_population[i].copy()
                        best_fitness = new_fitness[i]

        return best_solution, best_fitness
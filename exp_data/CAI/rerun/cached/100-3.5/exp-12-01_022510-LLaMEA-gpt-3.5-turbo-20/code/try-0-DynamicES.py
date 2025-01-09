import numpy as np

class DynamicES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.sigma = 1.0
        self.mu = 5
        self.lambda_ = 20

    def mutate(self, x):
        return x + self.sigma * np.random.randn(self.dim)

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget // self.lambda_):
            offspring = [self.mutate(best_solution) for _ in range(self.lambda_)]
            offspring_fitness = [func(sol) for sol in offspring]
            
            best_offspring_idx = np.argmin(offspring_fitness)
            if offspring_fitness[best_offspring_idx] < best_fitness:
                best_solution = offspring[best_offspring_idx]
                best_fitness = offspring_fitness[best_offspring_idx]
            
            successes = np.sum(np.array(offspring_fitness) < best_fitness)
            self.sigma *= np.exp(1/np.sqrt(self.dim) * (successes / self.lambda_ - 0.2))
        
        return best_solution
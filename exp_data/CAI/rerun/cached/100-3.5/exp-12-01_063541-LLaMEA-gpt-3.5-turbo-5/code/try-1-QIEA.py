import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.dim):
                # Apply quantum gate operation
                for j in range(len(self.population)):
                    if np.random.rand() < 0.5:
                        self.population[j][i] = -self.population[j][i]
            
            # Evaluate fitness of individuals
            fitness = [func(ind) for ind in self.population]

            # Sort individuals based on fitness
            sorted_idx = np.argsort(fitness)
            self.population = self.population[sorted_idx]

        # Return the best individual after the budget is exhausted
        return self.population[0]
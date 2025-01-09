import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        best_solution = population[np.argmin([func(ind) for ind in population])]
        
        for _ in range(self.budget):
            for i in range(len(population)):
                beta = np.random.uniform(0, 1)
                j = np.random.randint(0, len(population))
                population[i] = beta * population[i] + (1 - beta) * population[j]
                population[i] = np.clip(population[i], -5.0, 5.0)
                
                if func(population[i]) < func(best_solution):
                    best_solution = population[i]
        
        return best_solution
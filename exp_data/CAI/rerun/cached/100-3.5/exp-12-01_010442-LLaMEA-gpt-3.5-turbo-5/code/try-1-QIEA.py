import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]
            
            for i in range(self.budget):
                if i != best_idx:
                    prob = 1 / (1 + np.exp(fitness[i] - fitness[best_idx]))
                    rand_num = np.random.rand()
                    if rand_num < prob:
                        population[i] = np.logical_xor(population[i], best_individual)
                        fitness[i] = func(population[i])
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        
        return best_solution, best_fitness
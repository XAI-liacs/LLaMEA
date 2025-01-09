import numpy as np

class EMOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.max_iter = budget // self.pop_size

    def __call__(self, func):
        lb, ub = -5.0, 5.0
        population = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        
        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                # Perform DE mutation
                mutant = population[i] + 0.8 * (population[np.random.choice(self.pop_size, 3, replace=False)] - population[i])
                mutant = np.clip(mutant, lb, ub)
                
                # Perform PSO update
                velocity = 0.5 * velocity + 2.0 * np.random.random(self.dim) * (best_solution - population[i]) + 2.0 * np.random.random(self.dim) * (mutant - population[i])
                population[i] += velocity
                population[i] = np.clip(population[i], lb, ub)
                
                # Update fitness values
                new_fitness = func(population[i])
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    if new_fitness < fitness[best_idx]:
                        best_idx = i
                        best_solution = population[i]
        
        return best_solution
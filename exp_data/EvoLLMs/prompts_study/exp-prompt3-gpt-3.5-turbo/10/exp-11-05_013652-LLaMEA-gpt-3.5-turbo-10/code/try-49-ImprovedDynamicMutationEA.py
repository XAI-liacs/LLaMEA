import numpy as np

class ImprovedDynamicMutationEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]
            candidates = np.random.choice(np.delete(self.population, best_idx, axis=0), size=2, replace=False)
            mutation = candidates[0] - candidates[1]
            new_solution = best_solution + np.random.uniform(0.1, 0.2) * mutation
            if func(new_solution) < fitness[best_idx]:
                self.population[best_idx] = new_solution
        return self.population[np.argmin([func(ind) for ind in self.population])]
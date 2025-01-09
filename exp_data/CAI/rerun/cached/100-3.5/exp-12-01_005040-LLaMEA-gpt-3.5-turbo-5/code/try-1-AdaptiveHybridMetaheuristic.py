import numpy as np

class AdaptiveHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.max_iter = budget // self.pop_size

    def __call__(self, func):
        lb, ub = -5.0, 5.0
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        for _ in range(self.max_iter):
            best_idx = np.argmin(fitness)
            best = pop[best_idx]
            
            for i in range(self.pop_size):
                if np.random.rand() < 0.5:
                    new_pop = pop + 0.1 * np.random.randn(self.pop_size, self.dim)
                else:
                    new_pop = pop + 0.1 * np.random.randn(self.pop_size, self.dim) + 0.5 * (best - pop[i]) * np.random.randn(self.dim)
                
                new_pop = np.clip(new_pop, lb, ub)
                new_fitness = np.array([func(ind) for ind in new_pop])
                
                if np.min(new_fitness) < fitness[i]:
                    pop[i] = new_pop[np.argmin(new_fitness)]
                    fitness[i] = np.min(new_fitness)
        
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx]
        best_fitness = fitness[best_idx]
        
        return best_solution, best_fitness
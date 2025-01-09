import numpy as np

class HWDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 10 * self.dim
        F = 0.5
        CR = 0.9
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        for _ in range(self.budget // pop_size):
            new_pop = np.zeros((pop_size, self.dim))
            for i in range(pop_size):
                idxs = np.random.choice(pop_size, 3, replace=False)
                mutant = pop[idxs[0]] + F * (pop[idxs[1]] - pop[idxs[2]])
                crossover = np.random.rand(self.dim) < CR
                new_pop[i] = np.where(crossover, mutant, pop[i])
                
            new_fitness = np.array([func(ind) for ind in new_pop])
            for i in range(pop_size):
                if new_fitness[i] < fitness[i]:
                    pop[i] = new_pop[i]
                    fitness[i] = new_fitness[i]
        
        best_idx = np.argmin(fitness)
        return pop[best_idx]
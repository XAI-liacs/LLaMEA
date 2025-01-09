import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim, pop_size=10, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f = f
        self.cr = cr

    def select_parents(self, population, target_idx):
        idxs = np.random.choice(len(population), 3, replace=False)
        idxs = idxs[idxs != target_idx]
        return population[idxs]

    def mutate(self, population, target_idx):
        parents = self.select_parents(population, target_idx)
        mutant = parents[0] + self.f * (parents[1] - parents[2])
        return np.clip(mutant, -5.0, 5.0)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.cr
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def optimize(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        best_solution = population[np.argmin([func(ind) for ind in population])]
        
        for _ in range(self.budget - self.pop_size):
            new_population = []
            for idx, target in enumerate(population):
                mutant = self.mutate(population, idx)
                trial = self.crossover(target, mutant)
                if func(trial) < func(target):
                    population[idx] = trial
                    if func(trial) < func(best_solution):
                        best_solution = trial
            
        return best_solution
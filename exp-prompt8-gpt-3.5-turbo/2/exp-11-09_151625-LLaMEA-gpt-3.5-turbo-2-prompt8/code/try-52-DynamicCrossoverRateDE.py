import numpy as np

class DynamicCrossoverRateDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.CR = 0.9  # Initial crossover rate
        self.F = 0.5

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            target = population[i]
            idxs = np.arange(self.budget)
            np.random.shuffle(idxs)
            a, b, c = population[np.random.choice(idxs[:3], 3, replace=False)]
            mutant = a + self.F * (b - c)
            crossover = np.random.rand(self.dim) < self.CR
            trial = np.where(crossover, mutant, target)
            
            trial_fitness = func(trial)
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if np.random.rand() < 0.1:
                    self.F = np.clip(np.random.normal(0.5, 0.1), 0.1, 0.9)
                
                # Dynamic adaptation of crossover rate
                if np.random.rand() < 0.1:
                    self.CR = max(0.1, self.CR - 0.02) if np.random.rand() < 0.5 else min(0.9, self.CR + 0.02)
                    
        best_idx = np.argmin(fitness)
        return population[best_idx]
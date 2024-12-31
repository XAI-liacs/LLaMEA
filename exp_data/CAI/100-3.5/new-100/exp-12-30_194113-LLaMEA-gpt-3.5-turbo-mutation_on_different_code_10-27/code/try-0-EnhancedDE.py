import numpy as np

class EnhancedDE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.CR = 0.5
        self.F = 0.8

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            target_idx = np.random.randint(self.budget)
            base_idx1, base_idx2, base_idx3 = np.random.choice(np.delete(np.arange(self.budget), target_idx), size=3, replace=False)
            
            mutant = population[base_idx1] + self.F * (population[base_idx2] - population[base_idx3])
            crossover_points = np.random.rand(self.dim) < self.CR
            trial = np.where(crossover_points, mutant, population[target_idx])
            
            f_trial = func(trial)
            if f_trial < fitness[target_idx]:
                population[target_idx] = trial
                fitness[target_idx] = f_trial
            
            if fitness[target_idx] < self.f_opt:
                self.f_opt = fitness[target_idx]
                self.x_opt = population[target_idx]
            
        return self.f_opt, self.x_opt
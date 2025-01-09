import numpy as np
from joblib import Parallel, delayed

class DifferentialEvolutionImproved:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.CR = 0.9  # Crossover rate
        self.F = 0.5   # Differential weight
        
    def _mutation(self, population, idxs):
        a, b, c = population[np.random.choice(idxs, 3, replace=False)]
        return np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
        
    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            mutants = Parallel(n_jobs=-1)(delayed(self._mutation)(population, [idx for idx in range(self.budget) if idx != j]) for j in range(self.budget))
            
            trial_pop = np.where(np.random.rand(self.budget, self.dim) < self.CR, mutants, population)
            f_trial_pop = np.array([func(trial) for trial in trial_pop])
            
            improved_idxs = np.where(f_trial_pop < fitness)
            fitness[improved_idxs] = f_trial_pop[improved_idxs]
            population[improved_idxs] = trial_pop[improved_idxs]
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.f_opt:
                self.f_opt = fitness[best_idx]
                self.x_opt = population[best_idx]

        return self.f_opt, self.x_opt
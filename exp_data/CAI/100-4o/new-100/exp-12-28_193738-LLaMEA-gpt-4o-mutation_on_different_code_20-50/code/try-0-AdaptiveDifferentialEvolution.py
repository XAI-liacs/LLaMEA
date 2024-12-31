import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= self.pop_size
        
        while self.budget > 0:
            for i in range(self.pop_size):
                if self.budget <= 0:
                    break
                
                # Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                
                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                self.budget -= 1
                
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    
                    # Adapt F and CR based on success
                    self.F = np.clip(self.F + 0.1 * (np.random.rand() - 0.5), 0.1, 1.0)
                    self.CR = np.clip(self.CR + 0.1 * (np.random.rand() - 0.5), 0.1, 1.0)

                # Update global best
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
        
        return self.f_opt, self.x_opt
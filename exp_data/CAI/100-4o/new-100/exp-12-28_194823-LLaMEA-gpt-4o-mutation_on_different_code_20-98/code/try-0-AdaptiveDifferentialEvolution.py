import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.CR = 0.9
        self.F_base = 0.8
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = np.array(list(zip(func.bounds.lb, func.bounds.ub)))
        pop = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        self.update_best(pop, fitness)
        
        F = np.full(self.population_size, self.F_base)
        
        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                a, b, c = pop[indices]
                mutant = np.clip(a + F[i] * (b - c), bounds[:, 0], bounds[:, 1])
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, pop[i])
                
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    F[i] = np.random.uniform(0.5, 1.0)  # adaptive F
                    self.update_best(pop, fitness)
        
        return self.f_opt, self.x_opt

    def update_best(self, pop, fitness):
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.f_opt:
            self.f_opt = fitness[min_idx]
            self.x_opt = pop[min_idx]
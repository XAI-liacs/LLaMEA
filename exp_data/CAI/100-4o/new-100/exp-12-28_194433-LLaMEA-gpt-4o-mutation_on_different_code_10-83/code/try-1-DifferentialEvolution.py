import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Mutation factor
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        for generation in range(self.budget - self.pop_size):
            adaptive_cr = self.CR - (self.CR * generation / self.budget)
            adaptive_f = self.F + (0.2 * (1 - generation / self.budget))

            for i in range(self.pop_size):
                # Mutation
                idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                mutant = pop[idxs[0]] + adaptive_f * (pop[idxs[1]] - pop[idxs[2]])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < adaptive_cr
                trial = np.where(cross_points, mutant, pop[i])
                
                # Selection
                f = func(trial)
                if f < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial

        return self.f_opt, self.x_opt
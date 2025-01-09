import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        evals = self.pop_size
        
        while evals < self.budget:
            for i in range(self.pop_size):
                # Mutation with enriched exploration
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                if np.random.rand() < 0.5: 
                    mutant = np.clip(a + self.F * (b - c), bounds[0], bounds[1])  
                else:
                    mutant = np.clip(self.x_opt + self.F * (b - c), bounds[0], bounds[1])
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                evals += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                
                # Enhanced self-adaptive mechanism
                if evals % (self.pop_size * 5) == 0:
                    f_mean = fitness.mean()
                    improvement = (f_mean - self.f_opt) / f_mean
                    if improvement > 0.05:
                        self.F = min(self.F + improvement, 1.0)
                        self.CR = max(self.CR - improvement, 0.1)
                    else:
                        self.F = max(self.F - improvement, 0.1)
                        self.CR = min(self.CR + improvement, 1.0)

        return self.f_opt, self.x_opt
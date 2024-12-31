import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20, cr=0.9, f=0.8):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.cr = cr  # Crossover probability
        self.f = f  # Differential weight
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget - self.pop_size):
            for j in range(self.pop_size):
                # Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), func.bounds.lb, func.bounds.ub)
                
                # Crossover
                trial = np.copy(population[j])
                crossover_points = np.random.rand(self.dim) < self.cr
                trial[crossover_points] = mutant[crossover_points]
                
                # Selection
                f_trial = func(trial)
                if f_trial < fitness[j]:
                    population[j] = trial
                    fitness[j] = f_trial
                    
                    # Update global best
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

            # Adaptive parameters
            self.cr = max(0.1, min(1.0, self.cr + np.random.normal(0, 0.1)))
            self.f = max(0.4, min(1.0, self.f + np.random.normal(0, 0.1)))

        return self.f_opt, self.x_opt
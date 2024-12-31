import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=100, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # scale factor
        self.CR = CR  # crossover probability
        self.f_opt = np.Inf
        self.x_opt = None
    
    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, pop)
        
        # Update best solution
        min_idx = np.argmin(fitness)
        self.f_opt = fitness[min_idx]
        self.x_opt = pop[min_idx]

        evaluations = self.pop_size

        # Main evolution loop
        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Mutation
                idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                a, b, c = pop[idxs]
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                
                if evaluations >= self.budget:
                    break
        
        return self.f_opt, self.x_opt
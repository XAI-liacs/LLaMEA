import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size
        successful_mutations = 0
        
        while evaluations < self.budget:
            for i in range(self.pop_size):
                indices = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                x1, x2, x3 = population[indices]

                # Mutation and crossover
                mutant = x1 + self.F * (x2 - x3)
                mutant = np.clip(mutant, bounds[0], bounds[1])
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    successful_mutations += 1  # Track successful mutations
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                if evaluations >= self.budget:
                    break
            
            if successful_mutations > 0:  # Adjust F and CR dynamically
                self.F = np.clip(self.F + 0.05, 0.4, 0.9)
                self.CR = np.clip(self.CR + 0.05, 0.4, 0.95)
            else:
                self.F = np.clip(self.F - 0.05, 0.4, 0.9)
                self.CR = np.clip(self.CR - 0.05, 0.4, 0.95)
            successful_mutations = 0  # Reset for next round

        return self.f_opt, self.x_opt
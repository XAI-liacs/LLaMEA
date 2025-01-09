import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size
        
        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Select three random individuals for mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                
                # Mutation
                mutant = np.clip(x0 + self.F * (x1 - x2), func.bounds.lb, func.bounds.ub)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Evaluate the trial individual
                f_trial = func(trial)
                evaluations += 1
                
                # Selection
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                
                # Adaptive CR and F
                if evaluations < 0.8 * self.budget and fitness[i] > np.median(fitness):
                    self.CR = np.random.uniform(0.5, 1.0)
                    self.F = np.random.uniform(0.4, 0.9)

            # Early stopping if no improvement
            if evaluations >= self.budget or np.abs(self.f_opt - np.min(fitness)) < 1e-8:
                break
        
        return self.f_opt, self.x_opt
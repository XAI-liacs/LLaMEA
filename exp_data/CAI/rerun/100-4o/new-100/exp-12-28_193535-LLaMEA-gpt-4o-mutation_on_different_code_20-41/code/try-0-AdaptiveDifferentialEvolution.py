import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.F = 0.5  # initial mutation factor
        self.CR = 0.9  # crossover probability

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size

        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Select three distinct individuals
                candidates = list(range(self.pop_size))
                candidates.remove(i)
                a, b, c = population[np.random.choice(candidates, 3, replace=False)]

                # Mutation and Crossover
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                    
            # Update mutation factor adaptively
            success_rate = sum(fitness < self.f_opt) / self.pop_size
            self.F = max(0.1, min(0.9, 0.5 + (success_rate - 0.5)))
        
        return self.f_opt, self.x_opt
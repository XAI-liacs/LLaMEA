import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        remaining_budget = self.budget - self.pop_size

        while remaining_budget > 0:
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                # Mutation
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Local search (simple refinement)
                local_trial = np.clip(trial + np.random.normal(0, 0.1, self.dim), func.bounds.lb, func.bounds.ub)
                if remaining_budget >= 2:
                    f_trial = func(trial)
                    f_local_trial = func(local_trial)
                    remaining_budget -= 2
                elif remaining_budget == 1:
                    f_trial = func(trial)
                    f_local_trial = np.Inf
                    remaining_budget -= 1
                else:
                    break

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                if f_local_trial < fitness[i]:
                    population[i] = local_trial
                    fitness[i] = f_local_trial

                if fitness[i] < self.f_opt:
                    self.f_opt = fitness[i]
                    self.x_opt = population[i]

                if remaining_budget <= 0:
                    break
        
        return self.f_opt, self.x_opt
import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None
        self.num_evals = 0
        
    def differential_evolution(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.num_evals += self.pop_size
        
        while self.num_evals < self.budget:
            for i in range(self.pop_size):
                # Mutation
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f = func(trial)
                self.num_evals += 1
                if f < fitness[i]:
                    fitness[i] = f
                    population[i] = trial

                # Update the best solution found
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial

                # Check if budget is exceeded
                if self.num_evals >= self.budget:
                    break

        return self.f_opt, self.x_opt

    def adaptive_restart(self, func):
        # If budget allows, restart with new population
        while self.num_evals < self.budget:
            self.f_opt = np.Inf
            self.x_opt = None
            self.differential_evolution(func)

    def __call__(self, func):
        self.adaptive_restart(func)
        return self.f_opt, self.x_opt
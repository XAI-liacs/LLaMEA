import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        bounds = (func.bounds.lb, func.bounds.ub)
        pop = np.random.uniform(*bounds, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        # Evaluate initial population
        for i in range(self.budget - self.pop_size):
            # Select target vector and three random vectors that are distinct
            indices = np.random.choice(self.pop_size, 3, replace=False)
            a, b, c = pop[indices]

            # Mutation and Crossover
            mutant = np.clip(a + self.F * (b - c), *bounds)
            crossover_mask = np.random.rand(self.dim) < self.CR
            trial = np.where(crossover_mask, mutant, pop[i % self.pop_size])

            # Selection
            f_trial = func(trial)
            if f_trial < fitness[i % self.pop_size]:
                pop[i % self.pop_size] = trial
                fitness[i % self.pop_size] = f_trial

            # Adaptive parameter control
            self.F = 0.4 + 0.1 * (fitness.std() / fitness.mean())  # Adapt based on diversity
            self.CR = 0.8 + 0.1 * np.cos(i / self.budget * np.pi)  # Adapt over iterations

            # Update best solution found
            if f_trial < self.f_opt:
                self.f_opt = f_trial
                self.x_opt = trial

        return self.f_opt, self.x_opt
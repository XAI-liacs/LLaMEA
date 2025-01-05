import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.initial_pop_size = min(20, budget // dim)
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.initial_pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for i in range(self.budget - self.initial_pop_size):
            current_pop_size = self.initial_pop_size - i // (self.budget // self.initial_pop_size)
            idx = np.arange(current_pop_size)
            for j in range(current_pop_size):
                # Mutation
                a, b, c = population[np.random.choice(idx[idx != j], 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), func.bounds.lb, func.bounds.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[j])

                # Selection
                f_trial = func(trial)
                if f_trial < fitness[j]:
                    fitness[j] = f_trial
                    population[j] = trial

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

            # Adjust parameters adaptively
            self.mutation_factor = 0.5 + 0.3 * np.cos(i / self.budget * np.pi)
            self.crossover_rate = 0.9 - 0.4 * np.sin(i / self.budget * np.pi)  # Changed strategy here

        return self.f_opt, self.x_opt
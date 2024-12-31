import numpy as np

class DifferentialEvolutionAMC:
    def __init__(self, budget=10000, dim=10, population_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5, 5)
        self.mutation_factor = np.random.uniform(0.5, 1.0, self.pop_size)
        self.crossover_prob = np.random.uniform(0.1, 0.9, self.pop_size)

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.pop_size

        while evals < self.budget:
            for i in range(self.pop_size):
                # Mutation
                idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                a, b, c = population[idxs]
                mutant = np.clip(a + self.mutation_factor[i] * (b - c), self.bounds[0], self.bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < self.crossover_prob[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                evals += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    self.mutation_factor[i] = 0.5 * self.mutation_factor[i] + 0.5 * np.random.uniform(0.5, 1.0)
                    self.crossover_prob[i] = 0.5 * self.crossover_prob[i] + 0.5 * np.random.uniform(0.1, 0.9)

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if evals >= self.budget:
                    break

        return self.f_opt, self.x_opt
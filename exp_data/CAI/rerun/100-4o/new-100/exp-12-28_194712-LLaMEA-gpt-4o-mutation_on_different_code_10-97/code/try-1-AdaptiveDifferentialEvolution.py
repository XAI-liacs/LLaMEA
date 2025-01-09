import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.cr = 0.9  # Crossover probability
        self.f = 0.8   # Differential weight
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize a population with random solutions
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        # Update best solution found
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        # Begin optimization loop
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation: select three distinct individuals
                indices = [index for index in range(self.population_size) if index != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Create the mutant vector
                mutant = np.clip(a + self.f * (b - c), func.bounds.lb, func.bounds.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):  # Ensure that at least one parameter is from the mutant
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

                    # Update best solution found
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                # Adaptive control of differential weight
                if f_trial < self.f_opt:
                    self.f = np.clip(self.f + 0.1, 0.4, 1.0)
                    self.population_size = np.clip(self.population_size + 1, 10, self.budget - evaluations)
                else:
                    self.f = np.clip(self.f - 0.01, 0.4, 1.0)
                    self.population_size = np.clip(self.population_size - 1, 10, self.budget - evaluations)

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt
import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.inf
        self.x_opt = None
        self.F = 0.5  # Initial mutation factor
        self.CR = 0.9  # Initial crossover probability

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size

        # Track the best solution
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        while num_evaluations < self.budget:
            for i in range(self.pop_size):
                if num_evaluations >= self.budget:
                    break

                # Mutation: select three distinct random vectors
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Create mutant vector
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True

                trial = np.where(crossover_mask, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                num_evaluations += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial

                    # Update best solution
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

            # Adapt parameters
            self.F = np.clip(self.F + np.random.normal(0, 0.1), 0.1, 1.0)
            self.CR = np.clip(self.CR + np.random.normal(0, 0.1), 0.1, 1.0)

        return self.f_opt, self.x_opt
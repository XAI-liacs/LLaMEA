import numpy as np

class HybridDE:
    def __init__(self, budget, dim, pop_size=None):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size if pop_size else 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.evaluations = 0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += self.pop_size

        best_idx = np.argmin(fitness)
        best_vector = population[best_idx]
        best_fitness = fitness[best_idx]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                v = population[indices[0]] + self.F * (population[indices[1]] - population[indices[2]])
                v = np.clip(v, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True
                u = np.where(crossover_mask, v, population[i])

                # Selection
                u_fitness = func(u)
                self.evaluations += 1

                if u_fitness < fitness[i]:
                    population[i] = u
                    fitness[i] = u_fitness

                    if u_fitness < best_fitness:
                        best_vector = u
                        best_fitness = u_fitness

                if self.evaluations >= self.budget:
                    break

            # Adaptive parameter control (e.g., F and CR adaptation)
            self.F = 0.5 + 0.1 * (np.random.rand() - 0.5)
            self.CR = 0.9 + 0.1 * (np.random.rand() - 0.5)

        return best_vector, best_fitness
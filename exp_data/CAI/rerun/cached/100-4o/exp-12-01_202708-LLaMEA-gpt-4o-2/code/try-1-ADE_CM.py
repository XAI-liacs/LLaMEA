import numpy as np

class ADE_CM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9
        self.memory = []
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Select three distinct random indices
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]

                # Mutation
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    self.memory.append((trial, trial_fitness))

                    # Adaptive Update of F and CR based on memory
                    if len(self.memory) > 10:
                        self.memory.pop(0)
                    self.F = np.mean([f for _, f in self.memory if f < np.median(fitness)])
                    self.CR = 0.9  # Could be dynamically adjusted similarly

                if evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx]
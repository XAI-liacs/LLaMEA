import numpy as np

class ChaoticAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.5
        self.CR = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.evaluations = 0

    def chaotic_sequence(self, x):
        r = 3.99  # Chaotic constant
        return r * x * (1 - x)

    def __call__(self, func):
        fitness = np.apply_along_axis(func, 1, self.population)
        self.evaluations += self.population_size

        best_idx = np.argmin(fitness)
        best_solution = self.population[best_idx]
        best_fitness = fitness[best_idx]

        chaotic_param = np.random.rand()

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                trial = np.copy(self.population[i])

                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        trial[j] = mutant[j]

                f_trial = func(trial)
                self.evaluations += 1

                if f_trial < fitness[i]:
                    self.population[i] = trial
                    fitness[i] = f_trial

                    if f_trial < best_fitness:
                        best_solution = trial
                        best_fitness = f_trial

            if self.evaluations >= self.budget:
                break

            chaotic_param = self.chaotic_sequence(chaotic_param)
            self.F = 0.4 + chaotic_param * 0.2
            self.CR = 0.8 + chaotic_param * 0.1

        return best_solution, best_fitness
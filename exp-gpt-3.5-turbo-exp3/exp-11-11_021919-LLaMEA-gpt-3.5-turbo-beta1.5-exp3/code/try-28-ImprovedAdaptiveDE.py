import numpy as np

class ImprovedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.F_min, self.F_max = 0.1, 0.9
        self.CR_min, self.CR_max = 0.1, 0.9

    def dynamic_population_size(self, t):
        return int(10 + 40 * (1 - np.exp(-t / 1000)))

    def __call__(self, func):
        t = 0
        population_size = self.dynamic_population_size(t)
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        F, CR = self.F_max, self.CR_max

        for _ in range(self.budget):
            for i in range(population_size):
                a, b, c = np.random.choice(population_size, 3, replace=False)
                while a == i or b == i or c == i:
                    a, b, c = np.random.choice(population_size, 3, replace=False)

                mutant = population[a] + F * (population[b] - population[c])
                mutant = np.clip(mutant, -5.0, 5.0)

                j_rand = np.random.randint(self.dim)
                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])
                trial[j_rand] = mutant[j_rand]

                ft = func(trial)
                if ft < fitness[i]:
                    fitness[i] = ft
                    population[i] = trial

                    if ft < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = ft

            F = self.F_min + (_ / self.budget) * (self.F_max - self.F_min)
            CR = self.CR_max - (_ / self.budget) * (self.CR_max - self.CR_min)

            t += 1
            population_size = self.dynamic_population_size(t)
            if population_size > population.shape[0]:
                new_inds = np.random.uniform(-5.0, 5.0, (population_size - population.shape[0], self.dim))
                population = np.vstack([population, new_inds])
                new_fitness = np.array([func(ind) for ind in new_inds])
                fitness = np.hstack([fitness, new_fitness])

        return best_solution
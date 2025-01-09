import numpy as np

class DEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.f = 0.5
        self.cr = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def rand_in_bounds(minimum, maximum):
            return minimum + (maximum - minimum) * np.random.rand(self.dim)

        def ensure_bounds(vec, minimum, maximum):
            vec[vec < minimum] = minimum
            vec[vec > maximum] = maximum
            return vec

        def cost_function(x):
            return func(x)

        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([cost_function(p) for p in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]

        for _ in range(self.budget):
            for i in range(self.population_size):
                candidate = population[i]
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                mutant = ensure_bounds(a + self.f * (b - c), self.lower_bound, self.upper_bound)
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, candidate)

                y = candidate + self.c1 * np.random.rand(self.dim) * (best - candidate) + self.c2 * np.random.rand(self.dim) * (trial - candidate)
                population[i] = ensure_bounds(y, self.lower_bound, self.upper_bound)

                if cost_function(population[i]) < fitness[i]:
                    fitness[i] = cost_function(population[i])
                    if fitness[i] < cost_function(best):
                        best = population[i]

        return best
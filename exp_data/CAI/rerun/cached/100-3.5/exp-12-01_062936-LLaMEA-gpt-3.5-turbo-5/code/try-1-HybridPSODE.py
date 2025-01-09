import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.num_iterations = budget // self.population_size

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def mutate(x, best, gbest, F=0.5):
            idxs = np.random.choice(self.population_size, size=3, replace=False)
            a, b, c = x[idxs]
            return np.clip(a + F * (b - c), -5.0, 5.0)

        def apply_boundaries(x):
            return np.clip(x, -5.0, 5.0)

        population = initialize_population()
        pbest = population.copy()
        fitness_pbest = np.array([func(ind) for ind in pbest])
        gbest_idx = np.argmin(fitness_pbest)
        gbest = pbest[gbest_idx]

        for _ in range(self.num_iterations):
            for i in range(self.population_size):
                new_candidate = mutate(population[i], pbest[i], gbest)
                new_candidate_fitness = func(new_candidate)
                if new_candidate_fitness < fitness_pbest[i]:
                    pbest[i] = new_candidate
                    fitness_pbest[i] = new_candidate_fitness

                if new_candidate_fitness < func(gbest):
                    gbest = new_candidate

                population[i] = apply_boundaries(new_candidate)

        return gbest
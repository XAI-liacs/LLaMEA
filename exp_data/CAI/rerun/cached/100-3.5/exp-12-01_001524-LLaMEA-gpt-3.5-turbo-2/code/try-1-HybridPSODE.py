import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.max_iter = budget // self.population_size
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        def random_initialization():
            return np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=self.dim)

        population = [random_initialization() for _ in range(self.population_size)]
        fitness = [func(individual) for individual in population]

        for _ in range(self.max_iter):
            for i in range(self.population_size):
                p_best = population[np.argmin(fitness)]
                g_best = population[np.argmin(fitness)]

                w = 0.5 + 0.2 * np.random.rand()
                c1 = 1.5 * np.random.rand()
                c2 = 1.5 * np.random.rand()

                v = w * population[i] + c1 * np.random.rand() * (p_best - population[i]) + c2 * np.random.rand() * (g_best - population[i])
                v = np.clip(v, self.bounds[0], self.bounds[1])

                population[i] = v
                fitness[i] = func(v)

        best_solution = population[np.argmin(fitness)]
        return best_solution
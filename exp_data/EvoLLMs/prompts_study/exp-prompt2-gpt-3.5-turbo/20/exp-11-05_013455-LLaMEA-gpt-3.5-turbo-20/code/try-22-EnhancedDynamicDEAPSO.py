import numpy as np

class EnhancedDynamicDEAPSO(DynamicDEAPSO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def levy_flight(self, x, beta=1.5, scale=0.1):
        levy = np.random.standard_t(beta, size=x.shape) * scale
        return np.clip(x + levy, -5.0, 5.0)

    def __call__(self, func):
        def levy_mutate(x, beta=1.5, scale=0.1):
            return np.clip(x + np.random.standard_t(beta, size=x.shape) * scale, -5.0, 5.0)

        def differential_evolution(population, fitness, best, f, cr):
            new_population = np.copy(population)
            for i in range(self.population_size):
                a, b, c = population[np.random.choice(range(self.population_size), 3, replace=False)]
                x_new = levy_mutate(population[i])
                if np.all(x_new == population[i]) or np.random.rand() < cr:
                    x_new = a + f * (b - c)
                fitness_new = func(x_new)
                if fitness_new < fitness[i]:
                    new_population[i] = x_new
                    fitness[i] = fitness_new
                    if fitness_new < best:
                        best = fitness_new
                if np.random.rand() < self.mutation_prob:
                    new_population[i] = self.levy_flight(new_population[i])
            return new_population, fitness, best

        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best = np.min(fitness)
        f = 0.9
        cr = 0.9

        for _ in range(self.max_iterations):
            population, fitness, best = differential_evolution(population, fitness, best, f, cr)
            f = max(0.1, f * 0.95)  # Adaptive mutation rate adjustment
            cr = max(0.1, cr * 0.95)  # Adaptive crossover rate adjustment
        return best
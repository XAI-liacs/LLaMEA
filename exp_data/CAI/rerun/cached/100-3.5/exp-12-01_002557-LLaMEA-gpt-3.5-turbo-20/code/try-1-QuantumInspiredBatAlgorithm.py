import numpy as np

class QuantumInspiredBatAlgorithm:
    def __init__(self, budget, dim, population_size=10, A=0.5, r=0.5, Qmin=0.0, Qmax=2.0):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.A = A
        self.r = r
        self.Qmin = Qmin
        self.Qmax = Qmax

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        for _ in range(self.budget):
            Q = np.random.uniform(self.Qmin, self.Qmax, self.population_size)
            v = np.random.uniform(-1, 1, (self.population_size, self.dim))
            new_population = population + (population - best_solution) * self.A + v * Q[:, None]

            for i in range(self.population_size):
                for j in range(self.dim):
                    if np.random.rand() > self.r:
                        new_population[i, j] = best_solution[j] + np.random.uniform(-1, 1) * 5.0

            new_population = np.clip(new_population, -5.0, 5.0)
            new_fitness = np.array([func(ind) for ind in new_population])

            for i in range(self.population_size):
                if new_fitness[i] < fitness[i] and np.random.rand() < Q[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            if np.min(fitness) < best_fitness:
                best_solution = population[np.argmin(fitness)]
                best_fitness = np.min(fitness)

        return best_solution
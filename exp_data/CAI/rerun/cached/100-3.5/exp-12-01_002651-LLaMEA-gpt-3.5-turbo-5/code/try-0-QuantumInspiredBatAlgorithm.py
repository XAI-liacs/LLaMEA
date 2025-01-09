import numpy as np

class QuantumInspiredBatAlgorithm:
    def __init__(self, budget, dim, population_size=10, A=0.5, r=0.5, alpha=0.9, gamma=0.9, fmin=0, fmax=2):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.A = A
        self.r = r
        self.alpha = alpha
        self.gamma = gamma
        self.fmin = fmin
        self.fmax = fmax

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)

        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))

        t = 0
        while t < self.budget:
            for i in range(self.population_size):
                if np.random.rand() > self.gamma:
                    frequencies = self.fmin + (self.fmax - self.fmin) * np.random.rand()
                    velocities[i] += (population[i] - best_solution) * frequencies
                else:
                    velocities[i] += (population[i] - best_solution) * self.alpha

                if np.random.rand() < self.A:
                    for j in range(self.dim):
                        if np.random.rand() < self.r:
                            population[i, j] = best_solution[j] + np.random.uniform(-1, 1)
                        else:
                            population[i, j] += velocities[i, j]

                    population[i] = np.clip(population[i], -5.0, 5.0)
                    fitness = func(population[i])
                    t += 1

                    if fitness < best_fitness:
                        best_solution = population[i]
                        best_fitness = fitness

                    if t >= self.budget:
                        break

        return best_solution
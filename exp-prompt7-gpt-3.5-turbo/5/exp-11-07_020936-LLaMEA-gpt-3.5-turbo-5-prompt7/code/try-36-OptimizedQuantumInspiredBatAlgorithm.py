import numpy as np

class OptimizedQuantumInspiredBatAlgorithm:
    def __init__(self, budget, dim, population_size=10, loudness=0.5, pulse_rate=0.5, alpha=0.9, gamma=0.9):
        self.budget, self.dim, self.population_size, self.loudness, self.pulse_rate, self.alpha, self.gamma = budget, dim, population_size, loudness, pulse_rate, alpha, gamma
        self.lower_bound, self.upper_bound = -5.0, 5.0

    def __call__(self, func):
        bats = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        best_solution, best_fitness = bats[0], func(bats[0])

        for _ in range(self.budget):
            for i in range(self.population_size):
                if np.random.rand() > self.pulse_rate:
                    frequencies = np.clip(best_solution + self.alpha * (bats[i] - best_solution), self.lower_bound, self.upper_bound)
                    velocities[i] += frequencies * self.gamma
                else:
                    velocities[i] = np.random.uniform(-1, 1, self.dim) * np.linalg.norm(velocities[i])

                bats[i] = np.clip(bats[i] + velocities[i], self.lower_bound, self.upper_bound)
                new_fitness = func(bats[i])

                if np.random.rand() < self.loudness and new_fitness < best_fitness:
                    best_solution, best_fitness = bats[i], new_fitness

        return best_solution
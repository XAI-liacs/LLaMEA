import numpy as np

class OptimizedEnhancedQuantumBatAlgorithm:
    def __init__(self, budget, dim, population_size=10, loudness=0.5, pulse_rate=0.5, alpha=0.9, gamma=0.9):
        self.budget, self.dim, self.population_size, self.loudness, self.pulse_rate, self.alpha, self.gamma = budget, dim, population_size, loudness, pulse_rate, alpha, gamma

    def __call__(self, func):
        bats = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        best_solution, best_fitness = bats[0].copy(), func(bats[0])
        rand_vals = np.random.rand(self.budget)
        alpha_bats = self.alpha * bats
        norm_velocities = np.linalg.norm(velocities, axis=1)

        for i in range(self.budget):
            for j in range(self.population_size):
                frequencies = best_solution + alpha_bats[j] - best_solution
                frequencies = np.clip(frequencies, -5.0, 5.0)
                velocities[j] += frequencies * self.gamma if rand_vals[i] > self.pulse_rate else np.where(norm_velocities, np.random.uniform(-1, 1, self.dim), np.random.uniform(-1, 1, self.dim))
                new_solution = np.clip(bats[j] + velocities[j], -5.0, 5.0)
                new_fitness = func(new_solution)
                if rand_vals[i] < self.loudness and new_fitness < best_fitness:
                    bats[j], best_solution, best_fitness = new_solution.copy(), new_solution.copy(), new_fitness

        return best_solution
import numpy as np

class QuantumInspiredBatAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.min_freq = 0.0
        self.max_freq = 2.0
        self.alpha = 0.9
        self.gamma = 0.9
        self.loudness = np.ones(self.pop_size)
        self.pulse_rate = np.random.rand(self.pop_size)
        self.q_alpha = 0.05  # Quantum step size

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_bat = population[np.argmin(fitness)]
        
        evaluations = self.pop_size

        while evaluations < self.budget:
            for i in range(self.pop_size):
                freq = self.min_freq + (self.max_freq - self.min_freq) * np.random.rand()
                velocities[i] += (population[i] - best_bat) * freq
                candidate = population[i] + velocities[i]

                if np.random.rand() > self.pulse_rate[i]:
                    candidate = best_bat + self.q_alpha * np.random.randn(self.dim)

                candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
                candidate_fitness = func(candidate)
                evaluations += 1

                if (candidate_fitness < fitness[i]) and (np.random.rand() < self.loudness[i]):
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    self.loudness[i] *= self.alpha
                    self.pulse_rate[i] *= (1 - np.exp(-self.gamma * evaluations / self.budget))

                if candidate_fitness < func(best_bat):
                    best_bat = candidate

                if evaluations >= self.budget:
                    break

        return best_bat
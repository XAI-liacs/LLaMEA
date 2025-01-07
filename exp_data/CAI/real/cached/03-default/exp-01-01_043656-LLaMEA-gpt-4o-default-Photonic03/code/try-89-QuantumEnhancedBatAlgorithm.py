import numpy as np

class QuantumEnhancedBatAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.fmin = 0
        self.fmax = 2
        self.initial_loudness = 1.0
        self.final_loudness = 0.1
        self.initial_pulse_rate = 0.5
        self.final_pulse_rate = 0.1
        self.quantum_factor_initial = 0.3
        self.quantum_factor_final = 0.1

    def quantum_update(self, position, best_position, eval_count):
        delta = np.random.rand(self.dim)
        lambda_factor = (eval_count / self.budget)
        quantum_factor = self.quantum_factor_initial * (1 - lambda_factor) + self.quantum_factor_final * lambda_factor
        new_position = position + quantum_factor * (best_position - position) * delta
        return new_position

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        bats = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        velocities = np.zeros_like(bats)
        loudness = np.ones(self.population_size) * self.initial_loudness
        pulse_rate = np.ones(self.population_size) * self.initial_pulse_rate

        fitness = np.array([func(ind) for ind in bats])
        best_bat = bats[np.argmin(fitness)]
        best_fitness = fitness.min()
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                beta = np.random.rand()
                frequency = self.fmin + (self.fmax - self.fmin) * beta
                velocities[i] = velocities[i] + (bats[i] - best_bat) * frequency
                candidate_bat = bats[i] + velocities[i]
                candidate_bat = np.clip(candidate_bat, bounds[:, 0], bounds[:, 1])

                if np.random.rand() > pulse_rate[i]:
                    candidate_bat = self.quantum_update(candidate_bat, best_bat, eval_count)
                    candidate_bat = np.clip(candidate_bat, bounds[:, 0], bounds[:, 1])

                candidate_fitness = func(candidate_bat)
                eval_count += 1

                if candidate_fitness <= fitness[i] and np.random.rand() < loudness[i]:
                    bats[i] = candidate_bat
                    fitness[i] = candidate_fitness
                    loudness[i] = max(self.final_loudness, loudness[i] * 0.9)
                    pulse_rate[i] = min(self.final_pulse_rate, pulse_rate[i] * 1.1)

                    if candidate_fitness < best_fitness:
                        best_bat = candidate_bat
                        best_fitness = candidate_fitness

                if eval_count >= self.budget:
                    break

        return best_bat
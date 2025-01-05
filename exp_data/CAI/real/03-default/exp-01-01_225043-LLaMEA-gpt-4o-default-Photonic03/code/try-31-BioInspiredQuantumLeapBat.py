import numpy as np

class BioInspiredQuantumLeapBat:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.bats = np.random.uniform(size=(self.population_size, dim))
        self.velocities = np.zeros((self.population_size, dim))
        self.frequencies = np.random.uniform(0, 1, self.population_size)
        self.loudness = np.ones(self.population_size)
        self.pulse_rate = np.random.rand(self.population_size)
        self.current_best = self.bats[0].copy()
        self.current_best_fitness = np.inf
        self.fitness_evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        while self.fitness_evaluations < self.budget:
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                # Update frequency, velocity, and position
                self.frequencies[i] = np.random.uniform(0, 1)
                self.velocities[i] += (self.bats[i] - self.current_best) * self.frequencies[i]
                self.bats[i] += self.velocities[i]
                self.bats[i] = np.clip(self.bats[i], lower_bound, upper_bound)

                # Generate new solution by flying around current best
                if np.random.rand() > self.pulse_rate[i]:
                    self.bats[i] = self.current_best + 0.001 * np.random.randn(self.dim)

                # Evaluate new solutions
                fitness = func(self.bats[i])
                self.fitness_evaluations += 1

                # Update if the solution improves
                if fitness < self.current_best_fitness and np.random.rand() < self.loudness[i]:
                    self.current_best = self.bats[i].copy()
                    self.current_best_fitness = fitness
                    self.loudness[i] = max(0.1, self.loudness[i] * 0.9)
                    self.pulse_rate[i] = min(1, self.pulse_rate[i] * 1.1)

                # Quantum leap strategy for exploration
                quantum_prob = 0.3 - 0.1 * (self.fitness_evaluations / self.budget)
                if np.random.rand() < quantum_prob:
                    quantum_jump = lower_bound + np.random.rand(self.dim) * (upper_bound - lower_bound)
                    trial_fitness = func(quantum_jump)
                    self.fitness_evaluations += 1
                    if trial_fitness < self.current_best_fitness:
                        self.current_best = quantum_jump.copy()
                        self.current_best_fitness = trial_fitness

        return self.current_best
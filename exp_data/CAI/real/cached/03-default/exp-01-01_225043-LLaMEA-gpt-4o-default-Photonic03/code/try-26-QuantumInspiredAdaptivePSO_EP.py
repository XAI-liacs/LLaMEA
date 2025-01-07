import numpy as np

class QuantumInspiredAdaptivePSO_EP:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.uniform(size=(self.population_size, dim))
        self.velocities = np.random.uniform(size=(self.population_size, dim)) * 0.1
        self.personal_best = self.particles.copy()
        self.global_best = None
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best_fitness = np.inf
        self.fitness_evaluations = 0
        self.energy_pool = np.random.uniform(size=self.population_size)
        self.energy_threshold = 0.5

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        inertia_weight_decay = 0.95
        cognitive_coeff = lambda evals: 2.0 - 1.5 * (evals / self.budget)
        social_coeff = lambda evals: 1.5 + 1.0 * (evals / self.budget)

        while self.fitness_evaluations < self.budget:
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                fitness = func(self.particles[i])
                self.fitness_evaluations += 1

                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.particles[i].copy()

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.particles[i].copy()

                if self.energy_pool[i] < self.energy_threshold:
                    quantum_jump = lower_bound + np.random.rand(self.dim) * (upper_bound - lower_bound)
                    self.particles[i] = quantum_jump
                    self.energy_pool[i] = 1.0

                self.energy_pool[i] *= inertia_weight_decay

            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                inertia_weight = 0.9 * (1 - self.fitness_evaluations / self.budget)
                r1, r2 = np.random.rand(), np.random.rand()
                cog_velocity = cognitive_coeff(self.fitness_evaluations) * r1 * (self.personal_best[i] - self.particles[i])
                soc_velocity = social_coeff(self.fitness_evaluations) * r2 * (self.global_best - self.particles[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + cog_velocity + soc_velocity
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], lower_bound, upper_bound)

            if np.random.rand() < 0.2:
                energy_transfer_indices = np.random.choice(self.population_size, 2, replace=False)
                i, j = energy_transfer_indices
                if self.personal_best_fitness[i] < self.personal_best_fitness[j]:
                    energy_transfer = 0.1 * (self.personal_best_fitness[j] - self.personal_best_fitness[i])
                    self.energy_pool[i] += energy_transfer
                    self.energy_pool[j] -= energy_transfer
                else:
                    energy_transfer = 0.1 * (self.personal_best_fitness[i] - self.personal_best_fitness[j])
                    self.energy_pool[j] += energy_transfer
                    self.energy_pool[i] -= energy_transfer

        return self.global_best
import numpy as np

class HybridQuantumInspiredPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.rand(self.population_size, dim)
        self.velocities = np.random.rand(self.population_size, dim) * 0.1
        self.personal_best = self.particles.copy()
        self.global_best = None
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best_fitness = np.inf
        self.fitness_evaluations = 0
        self.archive = []
        self.learning_rates = np.full((self.population_size, dim), 0.1)
        self.historical_memory = np.zeros((self.population_size, dim))

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        while self.fitness_evaluations < self.budget:
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                fitness = func(self.particles[i])
                self.fitness_evaluations += 1

                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.particles[i].copy()
                    self.historical_memory[i] = self.particles[i] - self.historical_memory[i]

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.particles[i].copy()

                quantum_jump_prob = 0.2 - 0.1 * (self.fitness_evaluations / self.budget)
                if np.random.rand() < quantum_jump_prob:
                    self.particles[i] = lower_bound + np.random.rand(self.dim) * (upper_bound - lower_bound)

                if np.random.rand() < 0.15 and len(self.archive) < self.population_size:
                    self.archive.append(self.particles[i].copy())
                elif len(self.archive) == self.population_size:
                    worst_idx = np.argmax(self.personal_best_fitness)
                    if fitness < self.personal_best_fitness[worst_idx]:
                        self.archive[worst_idx] = self.particles[i].copy()

            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                inertia_weight = 0.9 - 0.4 * (self.fitness_evaluations / self.budget)
                cognitive_coeff = 1.5 + 0.5 * np.random.rand()
                social_coeff = 1.5 + 0.5 * np.random.rand()

                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_velocity = cognitive_coeff * r1 * (self.personal_best[i] - self.particles[i])
                social_velocity = social_coeff * r2 * (self.global_best - self.particles[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity + self.learning_rates[i] * self.historical_memory[i]
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], lower_bound, upper_bound)

            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                if len(self.archive) > 3:
                    archive_indices = np.random.choice(len(self.archive), 3, replace=False)
                    a, b, c = self.archive[archive_indices[0]], self.archive[archive_indices[1]], self.archive[archive_indices[2]]
                else:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = self.particles[indices[0]], self.particles[indices[1]], self.particles[indices[2]]

                F = 0.5 + 0.5 * np.random.rand()
                mutant = np.clip(a + F * (b - c), lower_bound, upper_bound)
                crossover_rate = 0.8 - 0.3 * (self.fitness_evaluations / self.budget)
                crossover_indices = np.random.rand(self.dim) < crossover_rate
                trial = np.where(crossover_indices, mutant, self.particles[i])

                trial_fitness = func(trial)
                self.fitness_evaluations += 1

                if trial_fitness < self.personal_best_fitness[i]:
                    self.particles[i] = trial.copy()
                    self.personal_best[i] = trial.copy()
                    self.personal_best_fitness[i] = trial_fitness
                    if trial_fitness < self.global_best_fitness:
                        self.global_best_fitness = trial_fitness
                        self.global_best = trial.copy()

        return self.global_best
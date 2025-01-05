import numpy as np

class QuantumMultiSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 60
        self.swarms = 3
        self.particles_per_swarm = self.population_size // self.swarms
        self.particles = np.random.uniform(size=(self.population_size, dim))
        self.velocities = np.random.uniform(size=(self.population_size, dim)) * 0.1
        self.personal_best = self.particles.copy()
        self.global_best = None
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best_fitness = np.inf
        self.fitness_evaluations = 0
        self.archives = [[] for _ in range(self.swarms)]

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        adaptive_scale_factor = lambda evals: 0.5 + 0.3 * np.cos(2 * np.pi * evals / self.budget)
        adaptive_crossover_rate = lambda evals: 0.8 - 0.5 * (evals / self.budget)

        while self.fitness_evaluations < self.budget:
            for swarm in range(self.swarms):
                for i in range(swarm * self.particles_per_swarm, (swarm + 1) * self.particles_per_swarm):
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

                    quantum_jump_prob = 0.2 - 0.1 * (self.fitness_evaluations / self.budget)
                    if np.random.rand() < quantum_jump_prob:
                        quantum_exploration = lower_bound + np.random.rand(self.dim) * (upper_bound - lower_bound)
                        self.particles[i] = quantum_exploration

                    if np.random.rand() < 0.05 and len(self.archives[swarm]) < self.particles_per_swarm:
                        self.archives[swarm].append(self.particles[i].copy())
                    elif len(self.archives[swarm]) == self.particles_per_swarm:
                        worst_idx = np.argmax(self.personal_best_fitness[swarm * self.particles_per_swarm:(swarm + 1) * self.particles_per_swarm])
                        global_idx = swarm * self.particles_per_swarm + worst_idx
                        if fitness < self.personal_best_fitness[global_idx]:
                            self.archives[swarm][worst_idx] = self.particles[i].copy()

            for swarm in range(self.swarms):
                for i in range(swarm * self.particles_per_swarm, (swarm + 1) * self.particles_per_swarm):
                    if self.fitness_evaluations >= self.budget:
                        break

                    inertia_weight = 0.6 - 0.3 * (self.fitness_evaluations / self.budget)
                    cognitive_coeff = 1.7
                    social_coeff = 2.0
                    r1, r2 = np.random.rand(), np.random.rand()

                    cognitive_velocity = cognitive_coeff * r1 * (self.personal_best[i] - self.particles[i])
                    social_velocity = social_coeff * r2 * (self.global_best - self.particles[i])
                    self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
                    self.particles[i] += self.velocities[i]
                    self.particles[i] = np.clip(self.particles[i], lower_bound, upper_bound)

            for swarm in range(self.swarms):
                for i in range(swarm * self.particles_per_swarm, (swarm + 1) * self.particles_per_swarm):
                    if self.fitness_evaluations >= self.budget:
                        break

                    if len(self.archives[swarm]) > 3:
                        archive_indices = np.random.choice(len(self.archives[swarm]), 3, replace=False)
                        a, b, c = self.archives[swarm][archive_indices[0]], self.archives[swarm][archive_indices[1]], self.archives[swarm][archive_indices[2]]
                    else:
                        indices = np.random.choice(self.particles_per_swarm, 3, replace=False)
                        a, b, c = self.particles[swarm * self.particles_per_swarm + indices[0]], self.particles[swarm * self.particles_per_swarm + indices[1]], self.particles[swarm * self.particles_per_swarm + indices[2]]

                    F = adaptive_scale_factor(self.fitness_evaluations)
                    mutant = np.clip(a + F * (b - c), lower_bound, upper_bound)
                    crossover_rate = adaptive_crossover_rate(self.fitness_evaluations)
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
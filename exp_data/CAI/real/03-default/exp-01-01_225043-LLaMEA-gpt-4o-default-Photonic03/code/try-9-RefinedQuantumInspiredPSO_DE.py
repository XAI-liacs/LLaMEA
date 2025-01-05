import numpy as np

class RefinedQuantumInspiredPSO_DE:
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

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        while self.fitness_evaluations < self.budget:
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                # Evaluate fitness of the current particle
                fitness = func(self.particles[i])
                self.fitness_evaluations += 1

                # Update personal best
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.particles[i].copy()

                # Update global best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.particles[i].copy()
                    
                # Stochastic exploration with dynamic adjustment
                exploration_factor = 0.1 + 0.4 * (1 - self.fitness_evaluations / self.budget)
                stochastic_jump = np.random.rand(self.dim) < exploration_factor
                exploration_vector = lower_bound + np.random.rand(self.dim) * (upper_bound - lower_bound)
                self.particles[i] = np.where(stochastic_jump, exploration_vector, self.particles[i])

                # Enhanced archive management
                if np.random.rand() < 0.1:
                    if len(self.archive) < self.population_size:
                        self.archive.append(self.particles[i].copy())
                    else:
                        worst_index = np.argmax([func(indiv) for indiv in self.archive])
                        if fitness < func(self.archive[worst_index]):
                            self.archive[worst_index] = self.particles[i].copy()

            # Update particles using PSO principles with adaptive learning rates
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                inertia_weight = 0.7 + 0.3 * np.sin(np.pi * self.fitness_evaluations / self.budget)
                cognitive_coeff = 1.5 + 0.5 * np.cos(2 * np.pi * self.fitness_evaluations / self.budget)
                social_coeff = 1.5 - 0.5 * np.sin(2 * np.pi * self.fitness_evaluations / self.budget)
                r1, r2 = np.random.rand(), np.random.rand()

                cognitive_velocity = cognitive_coeff * r1 * (self.personal_best[i] - self.particles[i])
                social_velocity = social_coeff * r2 * (self.global_best - self.particles[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], lower_bound, upper_bound)

            # Differential Evolution with adaptive crossover and mutation
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
                crossover_rate = 0.9 - 0.2 * np.cos(np.pi * self.fitness_evaluations / self.budget)
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
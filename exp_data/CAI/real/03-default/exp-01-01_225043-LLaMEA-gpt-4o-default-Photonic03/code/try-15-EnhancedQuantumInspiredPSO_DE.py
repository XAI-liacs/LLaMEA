import numpy as np

class EnhancedQuantumInspiredPSO_DE:
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

                fitness = func(self.particles[i])
                self.fitness_evaluations += 1

                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.particles[i].copy()

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.particles[i].copy()

                quantum_exploration = lower_bound + np.random.rand(self.dim) * (upper_bound - lower_bound)
                quantum_jump_prob = 0.3 - 0.2 * (self.fitness_evaluations / self.budget)
                if np.random.rand() < quantum_jump_prob:
                    self.particles[i] = quantum_exploration

                self._adaptive_archive_update(fitness, self.particles[i])

            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                niches = self._niche_strategy(self.particles, i)
                social_coeff = 2.0 if niches else 1.5
                inertia_weight = 0.9 - 0.5 * (self.fitness_evaluations / self.budget)
                cognitive_coeff = 1.7
                learning_rate = 0.1 + 0.9 * (1 - self.fitness_evaluations / self.budget)
                r1, r2 = np.random.rand(), np.random.rand()

                cognitive_velocity = cognitive_coeff * r1 * (self.personal_best[i] - self.particles[i])
                social_velocity = social_coeff * r2 * (self.global_best - self.particles[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + learning_rate * (cognitive_velocity + social_velocity)
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], lower_bound, upper_bound)

            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                a, b, c = self._select_from_archive()
                F = 0.6 + 0.4 * np.random.rand()
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

    def _niche_strategy(self, particles, index):
        niche_radius = np.linalg.norm(np.ptp(particles, axis=0)) / 10
        distances = np.linalg.norm(particles - particles[index], axis=1)
        return np.sum(distances < niche_radius) > 3

    def _adaptive_archive_update(self, fitness, particle):
        if np.random.rand() < 0.2 and len(self.archive) < self.population_size:
            self.archive.append(particle.copy())
        elif len(self.archive) == self.population_size:
            worst_idx = np.argmax([func(p) for p in self.archive])
            if fitness < func(self.archive[worst_idx]):
                self.archive[worst_idx] = particle.copy()

    def _select_from_archive(self):
        if len(self.archive) > 3:
            archive_indices = np.random.choice(len(self.archive), 3, replace=False)
            return (self.archive[archive_indices[0]], 
                    self.archive[archive_indices[1]], 
                    self.archive[archive_indices[2]])
        else:
            indices = np.random.choice(self.population_size, 3, replace=False)
            return (self.particles[indices[0]], 
                    self.particles[indices[1]], 
                    self.particles[indices[2]])
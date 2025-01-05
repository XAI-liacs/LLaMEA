import numpy as np

class QuantumAdaptiveSwarmOptimizer:
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
        self.diversity_threshold = 0.2

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

            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                local_neighborhood = self._select_neighborhood(i)
                neighborhood_best = min(local_neighborhood, key=lambda x: func(self.particles[x]))

                inertia_weight = 0.5 + 0.3 * np.random.rand()
                cognitive_coeff = 1.5
                social_coeff = 1.8
                r1, r2 = np.random.rand(), np.random.rand()

                cognitive_velocity = cognitive_coeff * r1 * (self.personal_best[i] - self.particles[i])
                social_velocity = social_coeff * r2 * (self.particles[neighborhood_best] - self.particles[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], lower_bound, upper_bound)

            quantum_exploration_prob = self._calculate_diversity() * 0.1
            for i in range(self.population_size):
                if np.random.rand() < quantum_exploration_prob:
                    quantum_exploration = lower_bound + np.random.rand(self.dim) * (upper_bound - lower_bound)
                    self.particles[i] = quantum_exploration

        return self.global_best

    def _select_neighborhood(self, index):
        neighborhood_size = max(3, self.population_size // 10)
        return np.random.choice(self.population_size, neighborhood_size, replace=False)

    def _calculate_diversity(self):
        mean_particle = np.mean(self.particles, axis=0)
        diversity = np.mean(np.linalg.norm(self.particles - mean_particle, axis=1))
        return diversity / (np.linalg.norm(self.particles[0]) + 1e-12)
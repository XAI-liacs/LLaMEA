import numpy as np

class QuantumDynamicSwarmOptimization:
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

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        def adaptive_quantum_prob(evals):
            return 0.4 - 0.2 * (evals / self.budget)

        def adaptive_inertia_weight(evals):
            return 0.9 - 0.5 * (evals / self.budget)

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

                if np.random.rand() < adaptive_quantum_prob(self.fitness_evaluations):
                    neighborhood = np.random.choice(self.population_size, size=3, replace=False)
                    quantum_jump = np.mean(self.particles[neighborhood], axis=0)
                    self.particles[i] = lower_bound + np.random.rand(self.dim) * (quantum_jump - lower_bound)

            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                inertia_weight = adaptive_inertia_weight(self.fitness_evaluations)
                cognitive_coeff = 1.5
                social_coeff = 2.0
                r1, r2 = np.random.rand(), np.random.rand()

                cognitive_velocity = cognitive_coeff * r1 * (self.personal_best[i] - self.particles[i])
                social_velocity = social_coeff * r2 * (self.global_best - self.particles[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], lower_bound, upper_bound)

        return self.global_best
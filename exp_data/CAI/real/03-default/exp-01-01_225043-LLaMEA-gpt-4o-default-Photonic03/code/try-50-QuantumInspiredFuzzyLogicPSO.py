import numpy as np

class QuantumInspiredFuzzyLogicPSO:
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

    def fuzzy_logic_params(self, evals):
        # Fuzzy logic to adjust parameters dynamically
        progress = evals / self.budget
        inertia_weight = 0.9 - 0.5 * progress
        social_component = 1.5 + 0.5 * (1 - progress)
        cognitive_component = 1.5 + 0.5 * progress
        return inertia_weight, cognitive_component, social_component

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

                inertia_weight, cognitive_coeff, social_coeff = self.fuzzy_logic_params(self.fitness_evaluations)
                r1, r2 = np.random.rand(), np.random.rand()

                cognitive_velocity = cognitive_coeff * r1 * (self.personal_best[i] - self.particles[i])
                social_velocity = social_coeff * r2 * (self.global_best - self.particles[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], lower_bound, upper_bound)

                # Quantum-inspired exploration mechanism
                if np.random.rand() < 0.1:
                    quantum_jump = lower_bound + np.random.rand(self.dim) * (upper_bound - lower_bound)
                    self.particles[i] = quantum_jump

        return self.global_best
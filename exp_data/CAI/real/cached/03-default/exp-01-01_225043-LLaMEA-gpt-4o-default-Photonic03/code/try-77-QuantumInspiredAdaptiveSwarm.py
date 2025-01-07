import numpy as np

class QuantumInspiredAdaptiveSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.positions = np.random.uniform(size=(self.population_size, dim))
        self.velocities = np.random.uniform(size=(self.population_size, dim)) * 0.1
        self.personal_best = self.positions.copy()
        self.global_best = None
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best_fitness = np.inf
        self.fitness_evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        while self.fitness_evaluations < self.budget:
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                fitness = func(self.positions[i])
                self.fitness_evaluations += 1

                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.positions[i].copy()

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.positions[i].copy()

            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break
                
                inertia_weight = 0.5 + 0.4 * np.random.rand()
                cognitive_coeff = 1.5
                social_coeff = 1.8
                quantum_coeff = 0.1 * np.sin(2 * np.pi * self.fitness_evaluations / self.budget)

                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_velocity = cognitive_coeff * r1 * (self.personal_best[i] - self.positions[i])
                social_velocity = social_coeff * r2 * (self.global_best - self.positions[i])
                quantum_velocity = quantum_coeff * (np.random.uniform(lower_bound, upper_bound, self.dim) - self.positions[i])
                
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity + quantum_velocity
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], lower_bound, upper_bound)

            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                neighborhood_size = max(1, int(self.population_size * (1 - self.fitness_evaluations / self.budget)))
                neighbors = np.random.choice(self.population_size, neighborhood_size, replace=False)
                
                best_neighbor = min(neighbors, key=lambda idx: func(self.positions[idx]))
                if func(self.positions[best_neighbor]) < fitness:
                    self.positions[i] = self.positions[best_neighbor]

        return self.global_best
import numpy as np

class AQIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.inertia_weight = 0.5
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.population = np.random.rand(self.population_size, dim)
        self.velocities = np.random.rand(self.population_size, dim) * 0.1
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_fitness = np.inf

    def __call__(self, func):
        bounds = np.vstack((func.bounds.lb, func.bounds.ub)).T
        evaluations = 0

        def decode_position(position):
            return bounds[:, 0] + ((bounds[:, 1] - bounds[:, 0]) * position)

        def evaluate_position(decoded_position):
            nonlocal evaluations
            fitness = np.array([func(ind) for ind in decoded_position])
            evaluations += len(decoded_position)
            return fitness

        def update_velocities_and_positions():
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                quantum_component = np.random.normal(0, 1, self.dim)
                self.velocities[i] = (
                    self.inertia_weight * self.velocities[i] +
                    self.c1 * r1 * (self.personal_best_positions[i] - self.population[i]) +
                    self.c2 * r2 * (self.global_best_position - self.population[i]) +
                    quantum_component
                )
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], 0, 1)

        while evaluations < self.budget:
            decoded_population = decode_position(self.population)
            fitness = evaluate_position(decoded_population)

            for i in range(self.population_size):
                if fitness[i] < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness[i]
                    self.personal_best_positions[i] = self.population[i]
                    if fitness[i] < self.global_best_fitness:
                        self.global_best_fitness = fitness[i]
                        self.global_best_position = self.population[i]

            update_velocities_and_positions()

        best_solution = decode_position(self.global_best_position)
        return best_solution, self.global_best_fitness
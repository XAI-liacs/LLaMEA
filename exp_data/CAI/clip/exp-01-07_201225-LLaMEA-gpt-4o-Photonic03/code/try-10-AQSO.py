import numpy as np

class AQSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(25, 5 + dim)
        self.population = np.random.rand(self.population_size, dim)
        self.velocity = np.zeros_like(self.population)
        self.fitness = np.full(self.population_size, np.inf)
        self.best_solution = None
        self.best_fitness = np.inf
        self.personal_best = np.copy(self.population)
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.inertia_weight = 0.9  # Initial inertia weight
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient

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

        def quantum_inspired_update(position, velocity, best_individual, best_global):
            phi = np.random.rand(self.dim)
            return position + self.inertia_weight * velocity + phi * (best_individual - position) + (1 - phi) * (best_global - position)

        while evaluations < self.budget:
            decoded_population = decode_position(self.population)
            fitness = evaluate_position(decoded_population)

            for i in range(self.population_size):
                if fitness[i] < self.personal_best_fitness[i]:
                    self.personal_best[i] = self.population[i]
                    self.personal_best_fitness[i] = fitness[i]
                    if fitness[i] < self.best_fitness:
                        self.best_fitness = fitness[i]
                        self.best_solution = self.population[i]

            for i in range(self.population_size):
                self.velocity[i] = quantum_inspired_update(
                    self.population[i], 
                    self.velocity[i], 
                    self.personal_best[i], 
                    self.best_solution
                )
                self.population[i] += self.velocity[i]
                self.population[i] = np.clip(self.population[i], 0, 1)

            # Update inertia weight
            self.inertia_weight = 0.4 + 0.5 * (1 - evaluations / self.budget)

        best_solution = decode_position(self.best_solution)
        return best_solution, self.best_fitness
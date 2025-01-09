import numpy as np

class QPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(20, 5 + dim)
        self.population = np.random.rand(self.population_size, dim)
        self.velocity = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, np.inf)
        self.best_positions = self.population.copy()
        self.best_fitness = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_fitness = np.inf
        self.c1 = 2.0  # Cognitive parameter
        self.c2 = 2.0  # Social parameter
        self.w = 0.7   # Inertia weight

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

        def adaptive_velocity(particle, velocity, pbest, gbest):
            r1, r2 = np.random.rand(), np.random.rand()
            cognitive_component = self.c1 * r1 * (pbest - particle)
            social_component = self.c2 * r2 * (gbest - particle)
            return self.w * velocity + cognitive_component + social_component

        while evaluations < self.budget:
            decoded_population = decode_position(self.population)
            fitness = evaluate_position(decoded_population)

            for i in range(self.population_size):
                if fitness[i] < self.best_fitness[i]:
                    self.best_fitness[i] = fitness[i]
                    self.best_positions[i] = self.population[i]
                    if fitness[i] < self.global_best_fitness:
                        self.global_best_fitness = fitness[i]
                        self.global_best_position = self.population[i]

            for i in range(self.population_size):
                self.velocity[i] = adaptive_velocity(self.population[i], self.velocity[i],
                                                     self.best_positions[i], self.global_best_position)
                self.population[i] += self.velocity[i]
                self.population[i] = np.clip(self.population[i], 0, 1)

            evaluations += self.population_size

        best_solution = decode_position(self.global_best_position)
        return best_solution, self.global_best_fitness
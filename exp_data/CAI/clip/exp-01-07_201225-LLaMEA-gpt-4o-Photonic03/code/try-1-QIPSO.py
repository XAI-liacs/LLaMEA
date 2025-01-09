import numpy as np

class QIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.position = np.random.rand(self.population_size, dim)
        self.velocity = np.random.rand(self.population_size, dim) * 0.1
        self.personal_best_position = np.copy(self.position)
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

        while evaluations < self.budget:
            decoded_position = decode_position(self.position)
            fitness = evaluate_position(decoded_position)

            for i in range(self.population_size):
                if fitness[i] < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness[i]
                    self.personal_best_position[i] = self.position[i]
                
                if fitness[i] < self.global_best_fitness:
                    self.global_best_fitness = fitness[i]
                    self.global_best_position = self.position[i]

            inertia_weight = 0.7
            cognitive_component = 1.5
            social_component = 1.5

            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i]
                                    + cognitive_component * r1 * (self.personal_best_position[i] - self.position[i])
                                    + social_component * r2 * (self.global_best_position - self.position[i]))
                self.position[i] = np.clip(self.position[i] + self.velocity[i], 0, 1)

        best_solution = decode_position(self.global_best_position)
        return best_solution, self.global_best_fitness
import numpy as np

class AdaptiveParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.inertia_weight = 0.9  # Initial inertia weight
        self.inertia_weight_min = 0.4  # Minimum inertia weight
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = np.copy(personal_best_positions[global_best_index])
        global_best_score = personal_best_scores[global_best_index]
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    self.inertia_weight * velocities[i] +
                    self.c1 * r1 * (personal_best_positions[i] - population[i]) +
                    self.c2 * r2 * (global_best_position - population[i])
                )
                population[i] = np.clip(population[i] + velocities[i], lb, ub)
                score = func(population[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_positions[i] = population[i]
                    personal_best_scores[i] = score

                if score < global_best_score:
                    global_best_position = population[i]
                    global_best_score = score

                if evaluations >= self.budget:
                    break

            self.inertia_weight = max(self.inertia_weight_min, self.inertia_weight - 0.01)  # Dynamic update

        return global_best_position, global_best_score
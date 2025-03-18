import numpy as np

class HybridParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.inertia_weight = 0.9  # Adjusted initial inertia weight
        self.mutation_rate = 0.1
        self.bounds = None

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        self.bounds = func.bounds
        lb, ub = self.bounds.lb, self.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([float('inf')] * self.population_size)

        global_best_position = None
        global_best_score = float('inf')
        evals = 0

        while evals < self.budget:
            for i in range(self.population_size):
                score = func(population[i])
                evals += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = population[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = population[i]

            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_component = self.c1 * r1 * r1 * (personal_best_positions[i] - population[i])  # Non-linear cognitive component
                social_component = self.c2 * r2 * (global_best_position - population[i])
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 cognitive_component + social_component)
                population[i] += velocities[i]

                # Adaptive mutation
                if np.random.rand() < self.mutation_rate:
                    mutation_vector = np.random.normal(0, 0.1, self.dim)
                    population[i] += mutation_vector

                # Ensure bounds
                population[i] = np.clip(population[i], lb, ub)

        return global_best_position, global_best_score
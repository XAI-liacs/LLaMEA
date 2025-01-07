import numpy as np

class HybridPSOGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight = 0.6
        self.cognitive_coef = 1.5
        self.social_coef = 1.5
        self.mutation_rate = 0.1
        self.cross_prob = 0.7

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in personal_best_positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]

        evaluations = self.population_size

        while evaluations < self.budget:
            self.inertia_weight = 0.4 + 0.2 * (1 - evaluations / self.budget)
            self.cognitive_coef = 1.0 + 0.5 * (1 - evaluations / self.budget)
            self.social_coef = 1.0 + 0.5 * (evaluations / self.budget)
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_coef * r1 * (personal_best_positions[i] - population[i]) +
                                 self.social_coef * r2 * (global_best_position - population[i]))
                population[i] += velocities[i]
                population[i] = np.clip(population[i], lb, ub)

                score = func(population[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = population[i]

                    if score < personal_best_scores[global_best_index]:
                        global_best_index = i
                        global_best_position = personal_best_positions[global_best_index]

            self.mutation_rate = 0.1 * (1 - evaluations / self.budget)

            offspring = np.copy(population)
            for i in range(0, self.population_size, 2):
                if np.random.rand() < self.cross_prob:
                    parent1, parent2 = offspring[i], offspring[(i+1) % self.population_size]
                    cross_point = np.random.randint(1, self.dim)
                    if np.random.rand() < 0.5:  # Dynamic crossover decision
                        parent2 = 0.5 * (parent1 + parent2)
                    offspring[i, :cross_point], offspring[(i+1) % self.population_size, :cross_point] = parent2[:cross_point], parent1[:cross_point]

            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    mutation_dim = np.random.randint(self.dim)
                    offspring[i, mutation_dim] += np.random.uniform(lb[mutation_dim], ub[mutation_dim]) * 0.05  # Adjusted mutation strength

            if evaluations % 10 == 0:
                self.population_size = max(10, int(self.population_size * 0.95))

            population = offspring[:self.population_size]

        return global_best_position, personal_best_scores[global_best_index]
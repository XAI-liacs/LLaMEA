import numpy as np

class SwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.population_size = self.initial_population_size
        self.inertia_weight = 0.5
        self.social_weight = 1.5
        self.cognitive_weight = 1.5

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.population_size

        while evaluations < self.budget:
            self.population_size = self.initial_population_size + int(10 * evaluations / self.budget)  # Dynamic population
            for i in range(self.population_size):
                self.inertia_weight = 0.9 - 0.5 * (evaluations / self.budget)
                self.social_weight = 1.5 + 0.5 * (evaluations / self.budget)
                self.cognitive_weight = 1.5 - 0.5 * (evaluations / self.budget)
                
                r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
                
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 (self.cognitive_weight + r3) * r1 * (personal_best_positions[i] - population[i]) +
                                 self.social_weight * r2 * (global_best_position - population[i]))
                population[i] = np.clip(population[i] + velocities[i], lb, ub)

                score = func(population[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = population[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = population[i]

                if evaluations >= self.budget:
                    break

        return global_best_position, global_best_score
import numpy as np

class ImprovedPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.9
        self.min_inertia_weight = 0.4
        self.cognitive_coeff = 1.6
        self.social_coeff = 1.5
        self.temperature = 1.0
        self.cooling_rate = 0.95
        self.elite_rate = 0.1  # Proportion of the elite group
        self.learning_rate_adapt = 0.1  # Adaptation rate for learning coefficients

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(x) for x in positions])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            elite_size = max(1, int(self.elite_rate * self.population_size))
            elite_indices = np.argsort(personal_best_scores)[:elite_size]

            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                # Adaptive learning coefficients
                self.cognitive_coeff += self.learning_rate_adapt * (np.random.rand() - 0.5)
                self.social_coeff += self.learning_rate_adapt * (np.random.rand() - 0.5)
                
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_coeff * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social_coeff * r2 * (global_best_position - positions[i]))

                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

                score = func(positions[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_scores[i] = score

                if score < global_best_score or np.random.rand() < np.exp((global_best_score - score) / self.temperature):
                    global_best_position = positions[i]
                    global_best_score = score

            # Elite selection strategy for potential new global best
            elite_best_idx = np.argmin(personal_best_scores[elite_indices])
            if personal_best_scores[elite_indices[elite_best_idx]] < global_best_score:
                global_best_position = personal_best_positions[elite_indices[elite_best_idx]]
                global_best_score = personal_best_scores[elite_indices[elite_best_idx]]

            self.inertia_weight = max(self.min_inertia_weight, self.inertia_weight - (0.5 / (self.budget / self.population_size)))
            self.temperature *= self.cooling_rate

        return global_best_position
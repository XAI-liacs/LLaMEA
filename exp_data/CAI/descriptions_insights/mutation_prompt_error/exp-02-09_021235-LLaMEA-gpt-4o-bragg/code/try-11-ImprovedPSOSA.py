import numpy as np

class ImprovedPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.9
        self.min_inertia_weight = 0.4
        self.max_inertia_weight = 1.2  # New: Adaptive inertia weight range
        self.cognitive_coeff = 1.6
        self.social_coeff = 1.5
        self.temperature = 1.0
        self.cooling_rate = 0.95
        self.neighborhood_size = 5
        self.alpha = 0.5  # New: Historical influence factor

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
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)

                # Dynamic neighborhood-based learning strategy
                neighbors_indices = np.random.choice(self.population_size, self.neighborhood_size, replace=False)
                local_best_idx = neighbors_indices[np.argmin(personal_best_scores[neighbors_indices])]
                local_best_position = personal_best_positions[local_best_idx]

                # New: Incorporate historical bests
                historical_best_position = self.alpha * personal_best_positions[i] + (1 - self.alpha) * global_best_position

                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_coeff * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social_coeff * r2 * (local_best_position - positions[i]) +
                                 self.alpha * (historical_best_position - positions[i]))  # New: historical influence

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

            self.inertia_weight = np.clip(self.inertia_weight - (0.5 / (self.budget / self.population_size)), self.min_inertia_weight, self.max_inertia_weight)  # Updated: Adaptive inertia adjustment
            self.temperature *= self.cooling_rate

            if evaluations < self.budget:
                global_best_idx = np.argmin(personal_best_scores)
                if personal_best_scores[global_best_idx] < global_best_score:
                    global_best_position = personal_best_positions[global_best_idx]
                    global_best_score = personal_best_scores[global_best_idx]

        return global_best_position
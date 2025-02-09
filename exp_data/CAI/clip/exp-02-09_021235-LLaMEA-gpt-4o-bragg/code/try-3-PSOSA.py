import numpy as np

class PSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.9  # Start with higher inertia weight
        self.min_inertia_weight = 0.4  # Allow adaptive reduction
        self.cognitive_coeff = 1.6  # Increased cognitive coefficient
        self.social_coeff = 1.5
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Adjusted cooling rate for better convergence

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initialize particle positions and velocities
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
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_coeff * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social_coeff * r2 * (global_best_position - positions[i]))

                # Update positions
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

                # Evaluate new position
                score = func(positions[i])
                evaluations += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_scores[i] = score

                # Apply simulated annealing acceptance criterion
                if score < global_best_score or np.random.rand() < np.exp((global_best_score - score) / self.temperature):
                    global_best_position = positions[i]
                    global_best_score = score

            # Adaptive inertia weight reduction
            self.inertia_weight = max(self.min_inertia_weight, self.inertia_weight - (0.5 / (self.budget / self.population_size)))

            # Cool down the temperature
            self.temperature *= self.cooling_rate

            # Update global best
            if evaluations < self.budget:
                global_best_idx = np.argmin(personal_best_scores)
                if personal_best_scores[global_best_idx] < global_best_score:
                    global_best_position = personal_best_positions[global_best_idx]
                    global_best_score = personal_best_scores[global_best_idx]

        return global_best_position
import numpy as np

class HybridPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.w = 0.5   # Inertia weight
        self.local_search_probability = 0.1

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initialize particles
        positions = np.random.uniform(low=lb, high=ub, size=(self.population_size, self.dim))
        velocities = np.random.uniform(low=-1, high=1, size=(self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(x) for x in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = func(global_best_position)

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Update velocities
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (personal_best_positions[i] - positions[i])
                    + self.c2 * r2 * (global_best_position - positions[i])
                )
                # Update positions
                positions[i] += velocities[i]

                # Boundary handling
                positions[i] = np.clip(positions[i], lb, ub)

                # Evaluate new position
                current_score = func(positions[i])
                evaluations += 1
                if evaluations >= self.budget:
                    break

                # Update personal best
                if current_score < personal_best_scores[i]:
                    personal_best_scores[i] = current_score
                    personal_best_positions[i] = positions[i]

                    # Update global best
                    if current_score < global_best_score:
                        global_best_score = current_score
                        global_best_position = positions[i]

            # Apply local search with a probability
            if np.random.rand() < self.local_search_probability:
                for i in range(self.population_size):
                    if evaluations < self.budget:
                        # Simple local search: random walk
                        candidate = positions[i] + np.random.normal(0, 0.1, self.dim)
                        candidate = np.clip(candidate, lb, ub)
                        candidate_score = func(candidate)
                        evaluations += 1
                        if candidate_score < personal_best_scores[i]:
                            personal_best_scores[i] = candidate_score
                            personal_best_positions[i] = candidate
                            if candidate_score < global_best_score:
                                global_best_score = candidate_score
                                global_best_position = candidate

        return global_best_position, global_best_score
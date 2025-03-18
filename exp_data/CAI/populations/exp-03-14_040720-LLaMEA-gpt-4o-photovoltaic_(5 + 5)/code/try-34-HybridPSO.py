import numpy as np

class HybridPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.w = 0.9   # Modified: Adaptive inertia weight
        self.local_search_probability = 0.1
        self.mutation_rate = 0.05  # New: Mutation rate for added diversity

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        positions = np.random.uniform(low=lb, high=ub, size=(self.population_size, self.dim))
        velocities = np.random.uniform(low=-1, high=1, size=(self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(x) for x in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]

        evaluations = self.population_size
        stagnation_counter = 0  # New: Tracking stagnation

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (personal_best_positions[i] - positions[i])
                    + self.c2 * r2 * (global_best_position - positions[i])
                )
                positions[i] += velocities[i]
                # Modified: Boundary reflection
                positions[i] = np.where(positions[i] < lb, lb + (lb - positions[i]), positions[i])
                positions[i] = np.where(positions[i] > ub, ub - (positions[i] - ub), positions[i])

                current_score = func(positions[i])
                evaluations += 1
                if evaluations >= self.budget:
                    break

                if current_score < personal_best_scores[i]:
                    personal_best_scores[i] = current_score
                    personal_best_positions[i] = positions[i]
                    if current_score < func(global_best_position):
                        global_best_position = positions[i]
                        stagnation_counter = 0  # New: Reset stagnation

            if np.random.rand() < self.local_search_probability:
                for i in range(self.population_size):
                    if evaluations < self.budget:
                        candidate = positions[i] + np.random.normal(0, 0.1, self.dim)
                        candidate = np.clip(candidate, lb, ub)
                        candidate_score = func(candidate)
                        evaluations += 1
                        if candidate_score < personal_best_scores[i]:
                            personal_best_scores[i] = candidate_score
                            personal_best_positions[i] = candidate
                            if candidate_score < func(global_best_position):
                                global_best_position = candidate

            # New: Mutation for diversity
            if np.random.rand() < self.mutation_rate:
                mutation_index = np.random.randint(self.population_size)
                positions[mutation_index] = np.random.uniform(low=lb, high=ub, size=self.dim)

            stagnation_counter += 1
            # New: Reinitialize if stagnation detected
            if stagnation_counter > 10:
                indices = np.random.choice(self.population_size, size=self.population_size // 2, replace=False)
                for idx in indices:
                    positions[idx] = np.random.uniform(low=lb, high=ub, size=self.dim)
                stagnation_counter = 0

        return global_best_position, func(global_best_position)
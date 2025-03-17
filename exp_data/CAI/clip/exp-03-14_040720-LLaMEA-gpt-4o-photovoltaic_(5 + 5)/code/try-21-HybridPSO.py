import numpy as np

class HybridPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.w = 0.9   # Modified: Initial inertia weight
        self.w_min = 0.4  # New: Minimum inertia weight
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
        global_best_score = np.min(personal_best_scores)  # New: Track global best score

        evaluations = self.population_size
        stagnation_counter = 0  # New: Tracking stagnation

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                self.w = self.w_min + (0.9 - self.w_min) * (1 - evaluations / self.budget)  # Modified: Adaptive inertia
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (personal_best_positions[i] - positions[i])
                    + self.c2 * r2 * (global_best_position - positions[i])
                )
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

                current_score = func(positions[i])
                evaluations += 1
                if evaluations >= self.budget:
                    break

                if current_score < personal_best_scores[i]:
                    personal_best_scores[i] = current_score
                    personal_best_positions[i] = positions[i]
                    if current_score < global_best_score:  # Modified: Use global best score
                        global_best_score = current_score  # Track new global best score
                        global_best_position = positions[i]
                        stagnation_counter = 0

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
                            if candidate_score < global_best_score:  # Modified: Use global best score
                                global_best_score = candidate_score
                                global_best_position = candidate

            if np.random.rand() < self.mutation_rate:
                mutation_index = np.random.randint(self.population_size)
                positions[mutation_index] = np.random.uniform(low=lb, high=ub, size=self.dim)

            stagnation_counter += 1
            if stagnation_counter > 10:
                elite_idx = np.argmin(personal_best_scores)  # New: Find elite index
                indices = np.random.choice(self.population_size, size=self.population_size // 2, replace=False)
                for idx in indices:
                    if idx != elite_idx:  # New: Preserve the elite solution
                        positions[idx] = np.random.uniform(low=lb, high=ub, size=self.dim)
                stagnation_counter = 0

        return global_best_position, func(global_best_position)
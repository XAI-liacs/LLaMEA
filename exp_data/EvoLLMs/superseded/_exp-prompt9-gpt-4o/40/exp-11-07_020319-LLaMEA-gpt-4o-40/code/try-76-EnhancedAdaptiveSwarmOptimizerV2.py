import numpy as np

class EnhancedAdaptiveSwarmOptimizerV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Increased population for more diversity
        self.inertia_weight = 0.9  # Adjusted inertia weight for better exploration
        self.c1_initial = 1.4
        self.c2_initial = 1.6
        self.c1_final = 0.6
        self.c2_final = 1.9
        self.mutation_factor = 0.85  # Changed mutation factor for variation
        self.recombination_rate = 0.8  # Lower recombination for focused crossover
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(ind) for ind in positions])
        global_best_position = positions[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)
        evaluations = self.population_size

        while evaluations < self.budget:
            inertia_weight = self.inertia_weight * (1 - evaluations / (2 * self.budget))
            c1 = self.c1_initial + (self.c1_final - self.c1_initial) * (evaluations / self.budget)
            c2 = self.c2_initial + (self.c2_final - self.c2_initial) * (evaluations / self.budget)

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (inertia_weight * velocities
                          + c1 * r1 * (personal_best_positions - positions)
                          + c2 * r2 * (global_best_position - positions))
            positions += velocities
            positions = np.clip(positions, self.lower_bound, self.upper_bound)

            scores = np.array([func(ind) for ind in positions])
            evaluations += self.population_size

            improved = scores < personal_best_scores
            personal_best_scores[improved] = scores[improved]
            personal_best_positions[improved] = positions[improved]

            min_score_idx = np.argmin(personal_best_scores)
            if personal_best_scores[min_score_idx] < global_best_score:
                global_best_score = personal_best_scores[min_score_idx]
                global_best_position = personal_best_positions[min_score_idx]

            if evaluations + self.population_size <= self.budget:
                for i in range(self.population_size):
                    indices = list(range(self.population_size))
                    indices.remove(i)
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant_vector = np.clip(personal_best_positions[a] 
                                            + self.mutation_factor * (personal_best_positions[b] - personal_best_positions[c]),
                                            self.lower_bound, self.upper_bound)
                    crossover_mask = np.random.rand(self.dim) < self.recombination_rate
                    trial_vector = np.where(crossover_mask, mutant_vector, positions[i])

                    trial_score = func(trial_vector)
                    evaluations += 1

                    if trial_score < personal_best_scores[i]:
                        personal_best_positions[i] = trial_vector
                        personal_best_scores[i] = trial_score
                        if trial_score < global_best_score:
                            global_best_score = trial_score
                            global_best_position = trial_vector

        return global_best_position, global_best_score
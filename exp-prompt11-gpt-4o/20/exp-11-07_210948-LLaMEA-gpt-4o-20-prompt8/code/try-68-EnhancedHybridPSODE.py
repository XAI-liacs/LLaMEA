import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 40
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.F = 0.5
        self.CR = 0.9

    def __call__(self, func):
        # Initialize particles and velocities using numpy operations
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.population_size, np.inf)

        # Evaluate initial particles
        scores = np.apply_along_axis(func, 1, positions)
        personal_best_scores = scores.copy()
        evaluations = self.population_size

        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        while evaluations < self.budget:
            r1, r2 = np.random.rand(2, self.population_size, self.dim)

            # Vectorized PSO Update
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))
            positions += velocities
            np.clip(positions, self.lower_bound, self.upper_bound, out=positions)

            # Evaluate new positions in vectorized form
            new_scores = np.apply_along_axis(func, 1, positions)
            evaluations += self.population_size

            # Update personal bests
            improved = new_scores < personal_best_scores
            personal_best_scores[improved] = new_scores[improved]
            personal_best_positions[improved] = positions[improved]

            # DE Mutation and Crossover
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                a, b, c = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                mutant = np.clip(positions[a] + self.F * (positions[b] - positions[c]), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, positions[i])
                trial_score = func(trial)
                evaluations += 1
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial

            # Update Global Best
            current_best_idx = np.argmin(personal_best_scores)
            if personal_best_scores[current_best_idx] < global_best_score:
                global_best_score = personal_best_scores[current_best_idx]
                global_best_position = personal_best_positions[current_best_idx]

        return global_best_position, global_best_score
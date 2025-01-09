import numpy as np

class HybridPSOwithADM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 50
        self.inertia = 0.9  # Initial inertia
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.mutation_factor = 0.5
        self.crossover_prob = 0.7
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.swarm_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def __call__(self, func):
        eval_count = 0
        iteration = 0
        while eval_count < self.budget:
            scores = np.apply_along_axis(func, 1, self.positions)
            eval_count += self.swarm_size

            # Update personal bests
            better_scores_idx = scores < self.personal_best_scores
            self.personal_best_scores[better_scores_idx] = scores[better_scores_idx]
            self.personal_best_positions[better_scores_idx] = self.positions[better_scores_idx]

            # Update global best
            min_score_idx = np.argmin(self.personal_best_scores)
            if self.personal_best_scores[min_score_idx] < self.global_best_score:
                self.global_best_score = self.personal_best_scores[min_score_idx]
                self.global_best_position = self.personal_best_positions[min_score_idx]

            # Update velocities and positions
            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            cognitive_term = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_term = self.social_coeff * r2 * (self.global_best_position - self.positions)
            # Self-adaptive inertia
            self.inertia = 0.9 - (0.5 * eval_count / self.budget)
            self.velocities = self.inertia * self.velocities + cognitive_term + social_term

            # Apply mutation on velocities
            for i in range(self.swarm_size):
                if np.random.rand() < self.crossover_prob:
                    idxs = np.random.choice(self.swarm_size, 3, replace=False)
                    mutant_vector = self.positions[idxs[0]] + self.mutation_factor * (self.positions[idxs[1]] - self.positions[idxs[2]])
                    mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                    self.velocities[i] = mutant_vector - self.positions[i]

            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score
import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20 + int(3 * np.log(dim))  # Dynamic swarm size based on dimensionality
        self.c1 = 1.49  # Cognitive parameter
        self.c2 = 1.49  # Social parameter
        self.w = 0.72   # Inertia weight
        self.mutation_factor = 0.5  # DE mutation factor
        self.recombination_rate = 0.9  # DE recombination rate
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Initialize particles
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, dim))
        self.p_best_positions = np.copy(self.positions)
        self.p_best_scores = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def __call__(self, func):
        evaluations = 0

        while evaluations < self.budget:
            # Evaluate current particles
            for i in range(self.pop_size):
                score = func(self.positions[i])
                if score < self.p_best_scores[i]:
                    self.p_best_scores[i] = score
                    self.p_best_positions[i] = self.positions[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

                evaluations += 1
                if evaluations >= self.budget:
                    break

            # Update velocities and positions using PSO formula
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            cognitive_component = self.c1 * r1 * (self.p_best_positions - self.positions)
            social_component = self.c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = self.w * self.velocities + cognitive_component + social_component

            # Update positions
            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Apply DE crossover for diversification
            for i in range(self.pop_size):
                a, b, c = np.random.choice(self.pop_size, 3, replace=False)
                mutant_vector = self.positions[a] + self.mutation_factor * (self.positions[b] - self.positions[c])
                trial_vector = np.where(np.random.rand(self.dim) < self.recombination_rate, mutant_vector, self.positions[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                trial_score = func(trial_vector)
                evaluations += 1
                if trial_score < self.p_best_scores[i]:
                    self.p_best_scores[i] = trial_score
                    self.p_best_positions[i] = trial_vector

                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial_vector

                if evaluations >= self.budget:
                    break

        return self.global_best_position, self.global_best_score
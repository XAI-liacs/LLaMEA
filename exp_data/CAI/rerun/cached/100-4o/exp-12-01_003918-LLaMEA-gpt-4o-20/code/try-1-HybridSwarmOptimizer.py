import numpy as np

class HybridSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(5, min(50, dim * 3))
        self.velocities = np.zeros((self.population_size, dim))
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.inertia_weight = 0.9
        self.eval_count = 0
        self.f = 0.5  # differential weight
        self.cr = 0.7  # crossover probability

    def __call__(self, func):
        while self.eval_count < self.budget:
            # Evaluate current positions and update personal and global bests
            for i in range(self.population_size):
                if self.eval_count < self.budget:
                    score = func(self.positions[i])
                    self.eval_count += 1
                    if score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = score
                        self.personal_best_positions[i] = self.positions[i]

                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = self.positions[i]

            # Differential Evolution Mutation for better exploration
            for i in range(self.population_size):
                if self.eval_count < self.budget:
                    indices = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant = self.positions[a] + self.f * (self.positions[b] - self.positions[c])
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                    crossover = np.random.rand(self.dim) < self.cr
                    trial = np.where(crossover, mutant, self.positions[i])
                    
                    trial_score = func(trial)
                    self.eval_count += 1
                    
                    if trial_score < self.personal_best_scores[i]:
                        self.positions[i] = trial
                        self.personal_best_scores[i] = trial_score
                        self.personal_best_positions[i] = trial

            # Update velocities and positions
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            cognitive_velocity = self.c1 * r1 * (self.personal_best_positions - self.positions)
            social_velocity = self.c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = self.inertia_weight * self.velocities + cognitive_velocity + social_velocity

            # Clamp velocities to a reasonable range
            v_max = 0.5 * (self.upper_bound - self.lower_bound)
            self.velocities = np.clip(self.velocities, -v_max, v_max)

            # Update positions and ensure they remain within bounds
            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Adjust inertia weight dynamically
            self.inertia_weight = 0.9 - (0.5 * (self.eval_count / self.budget))

        return self.global_best_position, self.global_best_score
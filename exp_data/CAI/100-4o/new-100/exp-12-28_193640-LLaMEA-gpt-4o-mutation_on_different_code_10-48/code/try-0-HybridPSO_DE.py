import numpy as np

class HybridPSO_DE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = min(100, self.budget // 10)
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.positions = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.Inf)
        self.global_best_position = None
        self.global_best_score = np.Inf
        self.fitness_evaluations = 0

    def __call__(self, func):
        while self.fitness_evaluations < self.budget:
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                # Evaluate fitness
                score = func(self.positions[i])
                self.fitness_evaluations += 1

                # Update personal and global bests
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

            # PSO update
            w = 0.5  # inertia weight
            c1 = 1.5  # cognitive parameter
            c2 = 1.5  # social parameter
            r1, r2 = np.random.rand(2)

            self.velocities = (w * self.velocities +
                               c1 * r1 * (self.personal_best_positions - self.positions) +
                               c2 * r2 * (self.global_best_position - self.positions))
            self.positions = self.positions + self.velocities
            self.positions = np.clip(self.positions, -5.0, 5.0)

            # DE mutation and crossover
            F = 0.8
            CR = 0.9
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.positions[indices]
                mutant = np.clip(x0 + F * (x1 - x2), -5.0, 5.0)

                trial = np.copy(self.positions[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                trial_score = func(trial)
                self.fitness_evaluations += 1

                if trial_score < self.personal_best_scores[i]:
                    self.positions[i] = trial
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial

        self.f_opt = self.global_best_score
        self.x_opt = self.global_best_position
        return self.f_opt, self.x_opt
import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 40
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.positions = np.random.uniform(-5, 5, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.global_best_position = np.copy(self.positions[0])
        self.personal_best_scores = np.full(self.num_particles, np.inf)
        self.global_best_score = np.inf
        self.F = 0.8  # DE mutation factor
        self.CR = 0.9  # DE crossover rate

    def __call__(self, func):
        eval_count = 0

        while eval_count < self.budget:
            # Evaluate current positions
            for i in range(self.num_particles):
                score = func(self.positions[i])
                eval_count += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = np.copy(self.positions[i])

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = np.copy(self.positions[i])

                if eval_count >= self.budget:
                    break

            # Update velocities and positions using PSO
            w = 0.5 + np.random.rand() / 2
            c1 = 1.5
            c2 = 1.5
            r1, r2 = np.random.rand(2, self.num_particles, self.dim)

            self.velocities = (w * self.velocities +
                               c1 * r1 * (self.personal_best_positions - self.positions) +
                               c2 * r2 * (self.global_best_position - self.positions))
            self.positions += self.velocities
            self.positions = np.clip(self.positions, -5.0, 5.0)

            # Differential Evolution step on best particles
            if eval_count < self.budget:
                for i in range(self.num_particles):
                    indices = list(range(self.num_particles))
                    indices.remove(i)
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant = np.clip(self.positions[a] + self.F * (self.positions[b] - self.positions[c]), -5.0, 5.0)
                    
                    trial = np.copy(self.positions[i])
                    for d in range(self.dim):
                        if np.random.rand() <= self.CR:
                            trial[d] = mutant[d]

                    trial_score = func(trial)
                    eval_count += 1

                    if trial_score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = trial_score
                        self.personal_best_positions[i] = trial

                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial

                    if eval_count >= self.budget:
                        break
        return self.global_best_position, self.global_best_score
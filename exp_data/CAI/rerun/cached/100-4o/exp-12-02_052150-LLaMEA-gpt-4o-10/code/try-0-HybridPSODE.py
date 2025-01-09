import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 40
        self.particles = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.CR = 0.9  # Crossover probability for DE
        self.F = 0.8   # Differential weight
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                # Evaluate particle
                if self.evaluations >= self.budget:
                    break
                score = func(self.particles[i])
                self.evaluations += 1

                # Update personal bests
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]

                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]

            # Update velocities and positions (PSO)
            for i in range(self.pop_size):
                inertia = self.velocities[i]
                cognitive = self.c1 * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.particles[i])
                social = self.c2 * np.random.rand(self.dim) * (self.global_best_position - self.particles[i])
                self.velocities[i] = inertia + cognitive + social
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], -5.0, 5.0)

            # Differential Evolution Mutation and Crossover
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break
                
                # Mutation: Choice of three distinct random particles
                indices = np.arange(self.pop_size)
                indices = indices[indices != i]
                a, b, c = self.particles[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), -5.0, 5.0)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.particles[i])

                # Selection
                if self.evaluations >= self.budget:
                    break
                trial_score = func(trial)
                self.evaluations += 1

                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial
                    self.particles[i] = trial

                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial

        return self.global_best_position, self.global_best_score
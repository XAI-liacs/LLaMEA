import numpy as np

class DEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 20
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best = self.population.copy()
        self.personal_best_scores = np.full(self.pop_size, np.inf)
        self.global_best = None
        self.global_best_score = np.inf
        self.F = 0.5  # DE scaling factor
        self.CR = 0.9  # DE crossover probability
        self.c1 = 2.0  # PSO cognitive component
        self.c2 = 2.0  # PSO social component
        self.w = 0.7  # PSO inertia weight

    def __call__(self, func):
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Differential Evolution mutation and crossover
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, self.population[i])

                # Evaluate trial vector
                trial_score = func(trial)
                evaluations += 1

                # Selection
                if trial_score < self.personal_best_scores[i]:
                    self.personal_best[i] = trial
                    self.personal_best_scores[i] = trial_score
                    if trial_score < self.global_best_score:
                        self.global_best = trial
                        self.global_best_score = trial_score

                # Particle Swarm Optimization velocity and position update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.personal_best[i] - self.population[i]) +
                                      self.c2 * r2 * (self.global_best - self.population[i]))
                self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lower_bound, self.upper_bound)

                if evaluations >= self.budget:
                    break

        return self.global_best, self.global_best_score
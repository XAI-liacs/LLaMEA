import numpy as np

class HybridPSOwithADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(40, budget // dim)
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.w = 0.5   # inertia weight
        self.f = 0.8   # DE mutation factor
        self.cr = 0.9  # DE crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            # Evaluate population
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                score = func(self.population[i])
                self.evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best[i] = self.population[i]
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = self.population[i]

            # Update velocities and positions using PSO
            r1, r2 = np.random.rand(2)
            self.velocities = (self.w * self.velocities
                               + self.c1 * r1 * (self.personal_best - self.population)
                               + self.c2 * r2 * (self.global_best - self.population))
            self.population += self.velocities

            # Apply DE mutation and crossover for exploration
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.population[a] + self.f * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                trial = np.copy(self.population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.cr:
                        trial[j] = mutant[j]
                score = func(trial)
                self.evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best[i] = trial
                    self.population[i] = trial
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best = trial
        return self.global_best
import numpy as np

class HybridPSODEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                score = func(self.population[i])
                self.evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.population[i]
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.population[i]
            
            # Update velocities and positions (PSO)
            w = 0.5
            c1 = 1.5
            c2 = 1.5
            r1, r2 = np.random.rand(2)
            self.velocities = (w * self.velocities +
                               c1 * r1 * (self.personal_best_positions - self.population) +
                               c2 * r2 * (self.global_best_position - self.population))
            self.population += self.velocities
            self.population = np.clip(self.population, self.lower_bound, self.upper_bound)

            # Differential Evolution mutation
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = self.population[np.random.choice(candidates, 3, replace=False)]
                mutant = np.clip(a + 0.8 * (b - c), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < 0.7, mutant, self.population[i])
                score = func(trial)
                self.evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.population[i] = trial
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = trial
        return self.global_best_position
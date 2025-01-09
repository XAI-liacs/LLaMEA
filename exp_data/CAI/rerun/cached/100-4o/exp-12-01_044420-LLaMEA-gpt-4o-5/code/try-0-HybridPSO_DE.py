import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 50
        self.w = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.F = 0.8
        self.CR = 0.9
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def __call__(self, func):
        eval_count = 0

        while eval_count < self.budget:
            # Evaluate fitness of the population
            scores = np.apply_along_axis(func, 1, self.population)
            eval_count += self.pop_size

            # Update personal bests
            better_mask = scores < self.personal_best_scores
            self.personal_best_scores[better_mask] = scores[better_mask]
            self.personal_best_positions[better_mask] = self.population[better_mask]

            # Update global best
            min_score_idx = np.argmin(self.personal_best_scores)
            if self.personal_best_scores[min_score_idx] < self.global_best_score:
                self.global_best_score = self.personal_best_scores[min_score_idx]
                self.global_best_position = self.personal_best_positions[min_score_idx]

            # Update velocities and positions using PSO
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            self.velocities = (self.w * self.velocities +
                               self.c1 * r1 * (self.personal_best_positions - self.population) +
                               self.c2 * r2 * (self.global_best_position - self.population))
            self.population += self.velocities
            self.population = np.clip(self.population, self.lb, self.ub)

            # Apply Differential Evolution crossover
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant_vector = a + self.F * (b - c)
                crossover = np.random.rand(self.dim) < self.CR
                trial_vector = np.where(crossover, mutant_vector, self.population[i])
                trial_vector = np.clip(trial_vector, self.lb, self.ub)
                trial_score = func(trial_vector)
                eval_count += 1
                if trial_score < scores[i]:
                    self.population[i] = trial_vector
                    scores[i] = trial_score
                    if trial_score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = trial_score
                        self.personal_best_positions[i] = trial_vector
                        if trial_score < self.global_best_score:
                            self.global_best_score = trial_score
                            self.global_best_position = trial_vector
                if eval_count >= self.budget:
                    break

        return self.global_best_position, self.global_best_score
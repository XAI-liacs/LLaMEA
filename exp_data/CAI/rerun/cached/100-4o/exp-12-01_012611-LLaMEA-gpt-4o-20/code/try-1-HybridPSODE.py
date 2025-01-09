import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.positions = np.random.uniform(-5, 5, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.w = 0.9  # inertia weight, increased for exploration
        self.c1 = 2.0  # cognitive coefficient, increased for personal influence
        self.c2 = 2.0  # social coefficient, increased for global influence
        self.mutation_factor = 0.9  # DE mutation factor, increased for diversity
        self.crossover_rate = 0.8  # DE crossover probability, slightly increased
        
    def __call__(self, func):
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                score = func(self.positions[i])
                evaluations += 1
                if evaluations >= self.budget:
                    break

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            for i in range(self.population_size):
                inertia = self.w * self.velocity[i]
                cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocity[i] = inertia + cognitive + social
                self.positions[i] = np.clip(self.positions[i] + self.velocity[i], -5, 5)

            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.positions[indices]
                mutant = np.clip(a + self.mutation_factor * (b - c), -5, 5)
                trial = np.array([
                    mutant[j] if np.random.rand() < self.crossover_rate else self.positions[i][j]
                    for j in range(self.dim)
                ])
                
                trial_score = func(trial)
                evaluations += 1
                if evaluations >= self.budget:
                    break

                if trial_score < self.personal_best_scores[i]:
                    self.positions[i] = trial
                    self.personal_best_scores[i] = trial_score
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial

        return self.global_best_position, self.global_best_score
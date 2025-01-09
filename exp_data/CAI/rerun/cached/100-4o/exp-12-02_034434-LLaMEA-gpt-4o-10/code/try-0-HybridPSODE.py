import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.global_best_position = None
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_score = np.inf

    def mutate_and_crossover(self, target_idx):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
        crossover = np.random.rand(self.dim) < self.crossover_rate
        trial = np.where(crossover, mutant, self.population[target_idx])
        return trial

    def update_particles(self, func):
        for i in range(self.population_size):
            score = func(self.population[i])
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.population[i]
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.population[i]

        for i in range(self.population_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            inertia = self.w * self.velocities[i]
            cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
            social = self.c2 * r2 * (self.global_best_position - self.population[i])
            self.velocities[i] = inertia + cognitive + social
            self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lower_bound, self.upper_bound)

    def __call__(self, func):
        eval_count = 0
        while eval_count < self.budget:
            self.update_particles(func)
            for i in range(self.population_size):
                trial = self.mutate_and_crossover(i)
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
        return self.global_best_position
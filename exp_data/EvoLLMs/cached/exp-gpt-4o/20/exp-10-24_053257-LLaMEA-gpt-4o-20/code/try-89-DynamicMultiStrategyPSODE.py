import numpy as np

class DynamicMultiStrategyPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(60, budget // (dim * 3))
        self.w = 0.3 + 0.4 * np.random.rand()  # inertia
        self.c1 = 1.5 + 0.1 * np.random.rand()  # cognitive
        self.c2 = 1.7 + 0.1 * np.random.rand()  # social
        self.F = 0.5 + 0.3 * np.random.rand()   # mutation
        self.CR = 0.6 + 0.3 * np.random.rand()  # crossover
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        learning_strategy = np.random.choice(['c1_adjust', 'c2_adjust', 'w_adjust', 'F_adjust'])
        
        while self.evaluations < self.budget:
            for i, solution in enumerate(self.population):
                if self.evaluations >= self.budget:
                    break
                score = func(solution)
                self.evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = solution
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = solution

            for i in range(self.population_size):
                r1, r2, r3 = np.random.rand(3)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.population[i])
                random_component = self.w * r3 * self.velocities[i]
                self.velocities[i] = random_component + cognitive_component + social_component
                self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lower_bound, self.upper_bound)

            if learning_strategy == 'c1_adjust':
                self.c1 = max(0.5, self.c1 - 0.05) if self.global_best_score < np.median(self.personal_best_scores) else min(2.5, self.c1 + 0.05)
            elif learning_strategy == 'c2_adjust':
                self.c2 = max(0.5, self.c2 - 0.05) if self.global_best_score < np.median(self.personal_best_scores) else min(2.5, self.c2 + 0.05)
            elif learning_strategy == 'w_adjust':
                self.w = max(0.3, self.w - 0.05) if self.global_best_score < np.median(self.personal_best_scores) else min(1.0, self.w + 0.05)
            elif learning_strategy == 'F_adjust':
                self.F = max(0.4, self.F - 0.05) if self.global_best_score < np.median(self.personal_best_scores) else min(0.9, self.F + 0.05)

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                indices = list(range(0, i)) + list(range(i + 1, self.population_size))
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutation_factor = self.F + (0.05 * np.random.randn())
                mutant_vector = self.population[a] + mutation_factor * (self.population[b] - self.population[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.CR, mutant_vector, self.population[i])
                trial_score = func(trial_vector)
                self.evaluations += 1
                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial_vector
                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial_vector

        return self.global_best_position, self.global_best_score
import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 40
        self.inertia_weight = 0.7
        self.cognitive_coef = 1.5
        self.social_coef = 1.5
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            # Evaluate current population
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

            # Update velocities and positions using PSO
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            self.inertia_weight = 0.9 - 0.5 * (self.evaluations / self.budget)  # Dynamic inertia weight
            for i in range(self.population_size):
                cognitive_component = self.cognitive_coef * r1 * (self.personal_best_positions[i] - self.population[i])
                social_component = self.social_coef * r2 * (self.global_best_position - self.population[i])
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_component + social_component)
                self.population[i] = np.clip(self.population[i] + self.velocities[i],
                                             self.lower_bound, self.upper_bound)

            # Apply DE mutation and crossover
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = np.clip(self.population[a] + self.mutation_factor * 
                                 (self.population[b] - self.population[c]),
                                 self.lower_bound, self.upper_bound)
                trial = np.copy(self.population[i])
                for d in range(self.dim):
                    if np.random.rand() < self.crossover_prob:
                        trial[d] = mutant[d]
                trial_score = func(trial)
                self.evaluations += 1
                if trial_score < self.personal_best_scores[i]:
                    self.population[i] = trial
                    self.personal_best_scores[i] = trial_score
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial

        return self.global_best_position, self.global_best_score
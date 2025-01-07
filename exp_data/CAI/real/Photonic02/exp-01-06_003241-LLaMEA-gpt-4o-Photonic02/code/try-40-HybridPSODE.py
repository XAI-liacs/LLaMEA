import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.inertia_weight = 0.7  # Increased inertia weight for better exploration
        self.cognitive_coefficient = 1.2  # Reduced for better balance
        self.social_coefficient = 1.8  # Increased for a stronger group influence
        self.mutation_factor = 0.8  # Dynamic adjustment (0.8 -> 0.5 + 0.3 * np.random.rand())
        self.crossover_rate = 0.7
        self.num_evaluations = 0
        self.elitism_rate = 0.1  # Rate for elitism to retain top performers

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.velocities = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.best_personal_positions = np.copy(self.population)
        self.best_personal_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.num_evaluations >= self.budget:
                break
            score = func(self.population[i])
            self.num_evaluations += 1
            if score < self.best_personal_scores[i]:
                self.best_personal_scores[i] = score
                self.best_personal_positions[i] = self.population[i]
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.population[i]

    def mutate_and_crossover(self, index):
        indices = [idx for idx in range(self.population_size) if idx != index]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant_vector = a + (0.5 + 0.3 * np.random.rand()) * (b - c)
        trial_vector = np.copy(self.population[index])
        
        for j in range(self.dim):
            if np.random.rand() < self.crossover_rate:
                trial_vector[j] = mutant_vector[j]
        return trial_vector

    def pso_update(self):
        elite_size = int(self.population_size * self.elitism_rate)
        elite_indices = np.argsort(self.best_personal_scores)[:elite_size]
        for i in range(self.population_size):
            if i in elite_indices:  # Preserve elite individuals
                continue
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                  self.cognitive_coefficient * r1 * (self.best_personal_positions[i] - self.population[i]) +
                                  self.social_coefficient * r2 * (self.global_best_position - self.population[i]))
            self.population[i] += self.velocities[i]

    def __call__(self, func):
        self.initialize_population(func.bounds)
        while self.num_evaluations < self.budget:
            self.evaluate_population(func)
            for i in range(self.population_size):
                if self.num_evaluations >= self.budget:
                    break                
                trial_vector = self.mutate_and_crossover(i)
                trial_score = func(trial_vector)
                self.num_evaluations += 1
                if trial_score < self.best_personal_scores[i]:
                    self.population[i] = trial_vector
                    self.best_personal_scores[i] = trial_score
                    self.best_personal_positions[i] = trial_vector
                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial_vector
            self.pso_update()
        return self.global_best_position, self.global_best_score
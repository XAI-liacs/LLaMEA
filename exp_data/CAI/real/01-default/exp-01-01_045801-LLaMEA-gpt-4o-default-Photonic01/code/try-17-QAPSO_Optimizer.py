import numpy as np

class QAPSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.initial_population_size = self.population_size
        self.alpha = 0.98  # Damping factor for convergence
        self.beta = 0.02  # Exploration factor
        self.population = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.personal_best_positions = self.population.copy()
        self.personal_best_scores = np.array([self.evaluate(ind) for ind in self.population])
        best_idx = np.argmin(self.personal_best_scores)
        self.global_best_position = self.population[best_idx].copy()
        self.global_best_score = self.personal_best_scores[best_idx]

    def evaluate(self, solution):
        return self.func(solution)

    def update_velocities_positions(self, lb, ub):
        for i in range(self.population_size):
            r1, r2 = np.random.rand(), np.random.rand()
            cognitive_component = self.alpha * r1 * (self.personal_best_positions[i] - self.population[i])
            social_component = self.alpha * r2 * (self.global_best_position - self.population[i])
            quantum_superposition = self.beta * np.random.uniform(lb, ub, self.dim)
            self.velocities[i] = cognitive_component + social_component + quantum_superposition
            self.population[i] = np.clip(self.population[i] + self.velocities[i], lb, ub)

    def update_personal_global_best(self):
        for i in range(self.population_size):
            score = self.evaluate(self.population[i])
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.population[i].copy()
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.population[i].copy()

    def resize_population(self):
        self.population_size = max(5, int(self.initial_population_size * (1 - self.evaluations / self.budget)))
        self.population = self.population[:self.population_size]
        self.velocities = self.velocities[:self.population_size]
        self.personal_best_positions = self.personal_best_positions[:self.population_size]
        self.personal_best_scores = self.personal_best_scores[:self.population_size]

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            self.resize_population()
            self.update_velocities_positions(lb, ub)
            self.update_personal_global_best()
            self.evaluations += self.population_size
            if self.evaluations >= self.budget:
                break

        return {'solution': self.global_best_position, 'fitness': self.global_best_score}
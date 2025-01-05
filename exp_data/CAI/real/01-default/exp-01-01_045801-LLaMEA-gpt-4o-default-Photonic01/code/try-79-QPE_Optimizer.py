import numpy as np

class QPE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.alpha = 0.5  # Influence of global best
        self.beta = 0.3   # Influence of individual best
        self.gamma = 0.2  # Random influence
        self.population = None
        self.velocities = None
        self.scores = None
        self.individual_best_positions = None
        self.individual_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.scores = np.array([self.evaluate(ind) for ind in self.population])
        self.individual_best_positions = self.population.copy()
        self.individual_best_scores = self.scores.copy()
        self.update_global_best()

    def evaluate(self, solution):
        return self.func(solution)

    def update_global_best(self):
        best_idx = np.argmin(self.scores)
        if self.scores[best_idx] < self.global_best_score:
            self.global_best_score = self.scores[best_idx]
            self.global_best_position = self.population[best_idx].copy()

    def update_positions_and_velocities(self, lb, ub):
        for i in range(self.population_size):
            r1, r2, r3 = np.random.rand(3, self.dim)
            cognitive_velocity = self.alpha * r1 * (self.individual_best_positions[i] - self.population[i])
            social_velocity = self.beta * r2 * (self.global_best_position - self.population[i])
            quantum_velocity = self.gamma * r3 * (np.random.uniform(lb, ub, self.dim) - self.population[i])
            self.velocities[i] = cognitive_velocity + social_velocity + quantum_velocity
            self.population[i] += self.velocities[i]
            self.population[i] = np.clip(self.population[i], lb, ub)
            current_score = self.evaluate(self.population[i])
            if current_score < self.individual_best_scores[i]:
                self.individual_best_scores[i] = current_score
                self.individual_best_positions[i] = self.population[i].copy()
                
            if self.evaluations < self.budget:
                self.evaluations += 1
            else:
                break
                
        self.update_global_best()

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            self.update_positions_and_velocities(lb, ub)

        return {'solution': self.global_best_position, 'fitness': self.global_best_score}
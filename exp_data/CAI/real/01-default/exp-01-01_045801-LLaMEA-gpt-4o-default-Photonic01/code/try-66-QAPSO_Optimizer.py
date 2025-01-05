import numpy as np

class QAPSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.alpha = 0.95  # Temperature decay factor
        self.temperature = 1.0  # Initial temperature
        self.velocity = np.zeros((self.population_size, self.dim))
        self.personal_best = None
        self.personal_best_scores = None
        self.global_best = None
        self.global_best_score = float('inf')
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.personal_best = self.population.copy()
        self.personal_best_scores = np.array([self.evaluate(ind) for ind in self.population])
        self.update_global_best()

    def evaluate(self, solution):
        return self.func(solution)

    def update_global_best(self):
        best_idx = np.argmin(self.personal_best_scores)
        if self.personal_best_scores[best_idx] < self.global_best_score:
            self.global_best_score = self.personal_best_scores[best_idx]
            self.global_best = self.personal_best[best_idx].copy()

    def quantum_annealing_update(self, lb, ub):
        for i in range(self.population_size):
            quantum_move = np.random.uniform(-1, 1, self.dim) * self.temperature
            new_position = self.personal_best[i] + quantum_move
            new_position = np.clip(new_position, lb, ub)
            new_score = self.evaluate(new_position)
            if new_score < self.personal_best_scores[i]:
                self.personal_best[i] = new_position
                self.personal_best_scores[i] = new_score
            self.velocity[i] = self.alpha * self.velocity[i] + new_position - self.population[i]
            self.population[i] += self.velocity[i]
            self.population[i] = np.clip(self.population[i], lb, ub)

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            self.quantum_annealing_update(lb, ub)
            self.temperature *= self.alpha  # Cool down the system
            self.update_global_best()
            self.evaluations += self.population_size

            if self.evaluations >= self.budget:
                break

        return {'solution': self.global_best, 'fitness': self.global_best_score}
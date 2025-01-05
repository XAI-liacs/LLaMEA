import numpy as np

class QIFA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.alpha = 0.5  # Attraction parameter
        self.beta_min = 0.2
        self.gamma = 1.0  # Light absorption coefficient
        self.q_factor = 0.1  # Quantum factor for superposition
        self.population = None
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.scores = np.array([self.evaluate(ind) for ind in self.population])
        self.update_best()

    def evaluate(self, solution):
        return self.func(solution)

    def update_best(self):
        best_idx = np.argmin(self.scores)
        self.best_solution = self.population[best_idx].copy()
        self.best_score = self.scores[best_idx]

    def move_fireflies(self, lb, ub):
        for i in range(self.population_size):
            for j in range(self.population_size):
                if self.scores[j] < self.scores[i]:
                    dist = np.linalg.norm(self.population[i] - self.population[j])
                    beta = (1 - self.beta_min) * np.exp(-self.gamma * dist ** 2) + self.beta_min
                    random_component = self.alpha * (np.random.rand(self.dim) - 0.5)
                    attraction = beta * (self.population[j] - self.population[i])
                    self.population[i] += attraction + random_component
                    self.population[i] = np.clip(self.population[i], lb, ub)
                    self.scores[i] = self.evaluate(self.population[i])
                    if self.scores[i] < self.best_score:
                        self.best_score = self.scores[i]
                        self.best_solution = self.population[i].copy()
                    self.evaluations += 1
                    if self.evaluations >= self.budget:
                        return

    def quantum_superposition(self, lb, ub):
        quantum_positions = self.best_solution + self.q_factor * (2 * np.random.rand(self.population_size, self.dim) - 1)
        quantum_positions = np.clip(quantum_positions, lb, ub)
        for i in range(self.population_size):
            quantum_score = self.evaluate(quantum_positions[i])
            if quantum_score < self.scores[i]:
                self.population[i] = quantum_positions[i]
                self.scores[i] = quantum_score
                if quantum_score < self.best_score:
                    self.best_score = quantum_score
                    self.best_solution = quantum_positions[i].copy()

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            self.move_fireflies(lb, ub)
            if self.evaluations < self.budget:
                self.quantum_superposition(lb, ub)

        return {'solution': self.best_solution, 'fitness': self.best_score}
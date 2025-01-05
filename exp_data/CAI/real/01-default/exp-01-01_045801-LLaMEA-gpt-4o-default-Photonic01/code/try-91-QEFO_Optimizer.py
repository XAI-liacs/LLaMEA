import numpy as np

class QEFO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.n_fireflies = 30
        self.alpha = 0.1  # Randomness coefficient
        self.beta_min = 0.2
        self.gamma = 1.0  # Absorption coefficient
        self.population = None
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.n_fireflies, self.dim))
        self.scores = np.array([self.evaluate(ind) for ind in self.population])
        self.update_best()

    def evaluate(self, solution):
        return self.func(solution)

    def move_fireflies(self, lb, ub):
        for i in range(self.n_fireflies):
            for j in range(self.n_fireflies):
                if self.scores[i] > self.scores[j]:
                    r = np.linalg.norm(self.population[i] - self.population[j])
                    beta = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * r**2)
                    attraction = beta * (self.population[j] - self.population[i])
                    randomization = self.alpha * np.random.uniform(-1, 1, self.dim)
                    entangled_influence = self.quantum_entanglement(self.population[i], self.population[j])
                    self.population[i] += attraction + randomization + entangled_influence
                    self.population[i] = np.clip(self.population[i], lb, ub)

    def quantum_entanglement(self, firefly_a, firefly_b):
        entanglement_strength = np.random.uniform(0.1, 0.5)
        quantum_state = np.random.choice([-1, 1], self.dim)
        entangled_influence = entanglement_strength * quantum_state * (firefly_b - firefly_a)
        return entangled_influence

    def update_best(self):
        best_idx = np.argmin(self.scores)
        self.best_solution = self.population[best_idx].copy()
        self.best_score = self.scores[best_idx]

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            self.move_fireflies(lb, ub)
            self.scores = np.array([self.evaluate(ind) for ind in self.population])
            self.update_best()
            self.evaluations += self.n_fireflies

        return {'solution': self.best_solution, 'fitness': self.best_score}
import numpy as np

class QuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha = 0.9
        self.beta = 0.5
        self.gamma = 0.2
        self.position = None
        self.best_position = None
        self.best_score = float('inf')
        self.c1 = 1.5
        self.c2 = 1.5

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.best_position = np.copy(self.position)
        self.best_scores = np.full(self.population_size, float('inf')) 

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i in range(self.population_size):
            if scores[i] < self.best_scores[i]:
                self.best_scores[i] = scores[i]
                self.best_position[i] = self.position[i]
            if scores[i] < self.best_score:
                self.best_score = scores[i]
                self.global_best = self.position[i]
        return scores

    def quantum_superposition(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        q_population = (self.alpha * self.best_position +
                        self.beta * self.global_best +
                        self.gamma * np.random.rand(self.population_size, self.dim))
        return np.clip(q_population, lb, ub)

    def update_position(self, q_population, scores, new_scores):
        for i in range(self.population_size):
            if new_scores[i] < scores[i]:
                self.position[i] = q_population[i]
                scores[i] = new_scores[i]

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            q_population = self.quantum_superposition(func.bounds)
            new_scores = np.array([func(p) for p in q_population])
            func_calls += self.population_size
            self.update_position(q_population, scores, new_scores)
        return self.global_best, self.best_score
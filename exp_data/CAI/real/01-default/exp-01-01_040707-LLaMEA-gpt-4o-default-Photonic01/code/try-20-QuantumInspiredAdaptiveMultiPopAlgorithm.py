import numpy as np

class QuantumInspiredAdaptiveMultiPopAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.position = None
        self.best_global_position = None
        self.best_global_score = float('inf')
        self.quantum_probability = 0.5
        self.tournament_size = 5

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.best_local_positions = np.copy(self.position)
        self.best_local_scores = np.full(self.population_size, float('inf'))

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i in range(self.population_size):
            if scores[i] < self.best_local_scores[i]:
                self.best_local_scores[i] = scores[i]
                self.best_local_positions[i] = self.position[i]
            if scores[i] < self.best_global_score:
                self.best_global_score = scores[i]
                self.best_global_position = self.position[i]
        return scores

    def quantum_superposition(self, position, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        mask = np.random.rand(self.population_size, self.dim) < self.quantum_probability
        new_positions = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        return np.where(mask, new_positions, position)

    def tournament_selection(self, scores):
        selected_indices = []
        for _ in range(self.population_size):
            candidates = np.random.choice(self.population_size, self.tournament_size, replace=False)
            best_candidate = candidates[np.argmin(scores[candidates])]
            selected_indices.append(best_candidate)
        return selected_indices

    def __call__(self, func):
        self.initialize(func.bounds)
        func_calls = 0
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            selected_indices = self.tournament_selection(scores)
            selected_positions = self.position[selected_indices]
            self.position = self.quantum_superposition(selected_positions, func.bounds)
        return self.best_global_position, self.best_global_score
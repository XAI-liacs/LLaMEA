import numpy as np

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.q_population = np.random.rand(self.population_size, self.dim)
        self.best_position = None
        self.best_score = np.inf
        self.alpha = 0.1  # Learning rate for rotation gate adjustments

    def _quantum_measurement(self, lb, ub):
        return lb + (ub - lb) * (self.q_population > np.random.rand(self.population_size, self.dim))

    def _update_quantum_gates(self, scores, lb, ub):
        best_indices = np.argsort(scores)[:5]  # Select top 5 solutions
        for i in best_indices:
            self.q_population[i] = self.alpha * self.best_position + (1 - self.alpha) * self.q_population[i]
        self.q_population = np.clip(self.q_population, lb, ub)

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub

        eval_count = 0
        while eval_count < self.budget:
            solutions = self._quantum_measurement(self.lb, self.ub)
            scores = np.zeros(self.population_size)

            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                scores[i] = func(solutions[i])
                eval_count += 1

                if scores[i] < self.best_score:
                    self.best_score = scores[i]
                    self.best_position = solutions[i]

            self._update_quantum_gates(scores, self.lb, self.ub)

        return self.best_position, self.best_score
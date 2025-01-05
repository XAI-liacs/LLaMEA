import numpy as np

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.alpha = 0.1  # Learning factor
        self.q_population = None
        self.observed_population = None
        self.best_solution = None
        self.best_score = np.inf

    def _initialize_population(self):
        # Quantum bit representation: each decision variable in a superposition of 0 and 1
        self.q_population = np.random.rand(self.population_size, self.dim, 2)
        self.q_population = self.q_population / np.linalg.norm(self.q_population, axis=2, keepdims=True)
        self.observed_population = np.zeros((self.population_size, self.dim))

    def _observe(self, lb, ub):
        # Collapse quantum bit to classical bit (0 or 1) using probabilistic observation
        for i in range(self.population_size):
            collapse = np.random.rand(self.dim) < self.q_population[i, :, 0]**2
            self.observed_population[i] = lb + (ub - lb) * collapse

    def _update_quantum_population(self, guide_position):
        for i in range(self.population_size):
            for j in range(self.dim):
                # Rotation gate for updating quantum bits
                theta = self.alpha * (2 * (self.observed_population[i, j] != guide_position[j]) - 1)
                rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                self.q_population[i, j] = self.q_population[i, j].dot(rotation_matrix)

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population()

        eval_count = 0
        while eval_count < self.budget:
            self._observe(self.lb, self.ub)
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                score = func(self.observed_population[i])
                eval_count += 1

                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = self.observed_population[i].copy()

            # Use the best solution found as the guiding position for quantum population update
            self._update_quantum_population(self.best_solution)

        return self.best_solution, self.best_score
import numpy as np

class QIEA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.population = None
        self.quantum_population = None
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        # Initialize quantum bits as probabilities
        self.quantum_population = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.population = self.quantum_to_real(self.quantum_population, lb, ub)
        self.scores = np.array([self.evaluate(ind) for ind in self.population])
        best_idx = np.argmin(self.scores)
        self.best_solution = self.population[best_idx].copy()
        self.best_score = self.scores[best_idx]

    def quantum_to_real(self, quantum_bits, lb, ub):
        # Convert quantum bits to real values within bounds
        return lb + (ub - lb) * quantum_bits

    def evaluate(self, solution):
        return self.func(solution)

    def quantum_rotation_gate(self, quantum_bit, best_bit):
        rotation_angle = np.pi / 10  # Fixed rotation angle
        delta = rotation_angle if quantum_bit < best_bit else -rotation_angle
        new_bit = quantum_bit + delta * (np.random.rand() - 0.5)
        return np.clip(new_bit, 0, 1)

    def evolve_quantum_population(self):
        # Apply quantum rotation gate based on best solution
        best_quantum = self.quantum_to_real(self.quantum_population, self.func.bounds.lb, self.func.bounds.ub)
        for i in range(self.population_size):
            for j in range(self.dim):
                self.quantum_population[i, j] = self.quantum_rotation_gate(self.quantum_population[i, j], best_quantum[j])

    def update_best(self):
        # Update the best solution found
        current_real_population = self.quantum_to_real(self.quantum_population, self.func.bounds.lb, self.func.bounds.ub)
        current_scores = np.array([self.evaluate(ind) for ind in current_real_population])
        for idx, score in enumerate(current_scores):
            if score < self.scores[idx]:
                self.scores[idx] = score
                self.population[idx] = current_real_population[idx].copy()
                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = current_real_population[idx].copy()

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            self.evolve_quantum_population()
            self.update_best()
            self.evaluations += self.population_size

        return {'solution': self.best_solution, 'fitness': self.best_score}
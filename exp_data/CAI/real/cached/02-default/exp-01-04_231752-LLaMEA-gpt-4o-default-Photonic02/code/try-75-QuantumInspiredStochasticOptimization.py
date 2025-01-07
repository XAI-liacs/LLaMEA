import numpy as np

class QuantumInspiredStochasticOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.quantum_bits = np.random.rand(self.population_size, self.dim)  # Quantum bit representation
        self.observed_states = np.zeros((self.population_size, self.dim))  # Observed states from quantum bits
        self.best_solution = None
        self.best_score = np.inf
        self.rotation_angle = np.pi / 4  # Rotation angle for quantum gates

    def _observe_states(self, lb, ub):
        """ Observe the quantum states to generate actual solution candidates """
        self.observed_states = lb + (ub - lb) * (self.quantum_bits > 0.5).astype(float)

    def _apply_quantum_rotation(self):
        """ Apply a quantum-inspired rotation gate to the quantum bits """
        for i in range(self.population_size):
            for d in range(self.dim):
                if np.random.rand() < 0.5:
                    # Quantum rotation
                    self.quantum_bits[i, d] = (self.quantum_bits[i, d] + self.rotation_angle) % 1.0
                else:
                    # Quantum rotation in the opposite direction
                    self.quantum_bits[i, d] = (self.quantum_bits[i, d] - self.rotation_angle) % 1.0

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        eval_count = 0

        while eval_count < self.budget:
            self._observe_states(self.lb, self.ub)
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                score = func(self.observed_states[i])
                eval_count += 1

                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = np.copy(self.observed_states[i])

            self._apply_quantum_rotation()

        return self.best_solution, self.best_score
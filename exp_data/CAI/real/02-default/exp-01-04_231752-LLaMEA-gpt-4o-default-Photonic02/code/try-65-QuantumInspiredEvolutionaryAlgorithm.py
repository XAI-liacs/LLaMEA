import numpy as np

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.q_population = np.random.rand(self.population_size, self.dim, 2)  # Quantum bits
        self.best_position = None
        self.best_score = np.inf

    def _initialize_population(self, lb, ub):
        self.p_real = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.best_position = np.copy(self.p_real[0])
    
    def _measure_population(self):
        # Convert quantum bits to real values
        return np.array([[(0 if np.random.rand() < q[0] else 1) for q in individual] for individual in self.q_population])

    def _update_population(self, lb, ub):
        # Apply quantum gates for updating
        for i in range(self.population_size):
            if np.random.rand() < 0.5:
                # Quantum NOT gate
                self.q_population[i] = 1 - self.q_population[i]
            else:
                # Quantum rotation
                rotation_angle = np.random.rand() * np.pi / 4
                self.q_population[i] = np.array([self._apply_rotation(q, rotation_angle) for q in self.q_population[i]])
            
            # Measure the new positions
            self.p_real[i] = lb + self._measure_population()[i] * (ub - lb)

    def _apply_rotation(self, q, angle):
        # Apply a quantum rotation gate
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        new_q0 = cos_angle * q[0] - sin_angle * q[1]
        new_q1 = sin_angle * q[0] + cos_angle * q[1]
        return np.array([new_q0, new_q1])

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                score = func(self.p_real[i])
                eval_count += 1

                if score < self.best_score:
                    self.best_score = score
                    self.best_position = self.p_real[i]

            self._update_population(self.lb, self.ub)

        return self.best_position, self.best_score
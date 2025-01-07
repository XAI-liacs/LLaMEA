import numpy as np

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.qubit_population = None
        self.best_solution = None
        self.best_score = np.inf
        self.rotation_angle = 0.05  # Rotation angle for quantum gates

    def _initialize_qubit_population(self):
        # Initialize qubit representation with equal probabilities
        self.qubit_population = np.full((self.population_size, self.dim, 2), 1/np.sqrt(2))

    def _measure_population(self, lb, ub):
        # Convert qubit representation to real values
        real_values = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            for j in range(self.dim):
                # Measure qubit and obtain real value
                prob_zero = self.qubit_population[i, j, 0]**2
                real_values[i, j] = lb[j] if np.random.rand() < prob_zero else ub[j]
        return real_values

    def _update_qubit_population(self, real_values, scores, lb, ub):
        # Update qubit states using a quantum-inspired rotation gate
        for i in range(self.population_size):
            if scores[i] < self.best_score:
                self.best_score = scores[i]
                self.best_solution = real_values[i]

            for j in range(self.dim):
                best_bit = (self.best_solution[j] - lb[j]) / (ub[j] - lb[j])
                current_bit = (real_values[i, j] - lb[j]) / (ub[j] - lb[j])
                # Calculate rotation direction and apply rotation
                rotation_direction = np.sign(best_bit - current_bit)
                theta = rotation_direction * self.rotation_angle
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)

                a, b = self.qubit_population[i, j, 0], self.qubit_population[i, j, 1]
                self.qubit_population[i, j, 0] = cos_theta * a - sin_theta * b
                self.qubit_population[i, j, 1] = sin_theta * a + cos_theta * b

                # Normalize qubits to ensure valid probability distribution
                norm = np.linalg.norm(self.qubit_population[i, j])
                self.qubit_population[i, j] /= norm

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_qubit_population()

        eval_count = 0
        while eval_count < self.budget:
            real_values = self._measure_population(self.lb, self.ub)

            scores = np.zeros(self.population_size)
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                scores[i] = func(real_values[i])
                eval_count += 1

            self._update_qubit_population(real_values, scores, self.lb, self.ub)

        return self.best_solution, self.best_score
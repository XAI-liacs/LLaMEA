import numpy as np

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.qubit_population = None
        self.observed_population = None
        self.best_solution = None
        self.best_score = np.inf
        self.alpha = 0.5  # Quantum rotation angle coefficient

    def _initialize_population(self):
        self.qubit_population = np.random.rand(self.population_size, self.dim, 2)
        self.qubit_population /= np.linalg.norm(self.qubit_population, axis=2, keepdims=True)
        self.observed_population = np.zeros((self.population_size, self.dim))

    def _observe_population(self):
        for i in range(self.population_size):
            for j in range(self.dim):
                prob_0 = self.qubit_population[i, j, 0]**2
                self.observed_population[i, j] = 0 if np.random.rand() < prob_0 else 1

    def _update_qubits(self, best_qubit):
        for i in range(self.population_size):
            for j in range(self.dim):
                delta_theta = self.alpha * (best_qubit[j] - self.qubit_population[i, j])
                rotation_matrix = np.array([[np.cos(delta_theta), -np.sin(delta_theta)],
                                            [np.sin(delta_theta), np.cos(delta_theta)]])
                self.qubit_population[i, j] = np.dot(rotation_matrix, self.qubit_population[i, j])
                self.qubit_population[i, j] /= np.linalg.norm(self.qubit_population[i, j])

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population()

        eval_count = 0
        while eval_count < self.budget:
            self._observe_population()
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                candidate_solution = self.lb + (self.ub - self.lb) * self.observed_population[i]
                score = func(candidate_solution)
                eval_count += 1

                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = candidate_solution

            best_qubit = self.qubit_population[np.argmin([func(self.lb + (self.ub - self.lb) * self.observed_population[i]) for i in range(self.population_size)])]
            self._update_qubits(best_qubit)

        return self.best_solution, self.best_score
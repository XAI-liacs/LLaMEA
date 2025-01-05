import numpy as np

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.qubit_population = None
        self.best_solution = None
        self.best_score = np.inf
        self.alpha = 0.1  # Quantum rotation angle

    def _initialize_population(self):
        # Initialize qubit population with random angles in radians
        self.qubit_population = np.random.uniform(0, 2 * np.pi, (self.population_size, self.dim))

    def _qubit_to_solution(self, qubit):
        # Map qubit states to real-valued solution using probability amplitudes
        return np.cos(qubit)**2

    def _evaluate_population(self, func):
        eval_count = 0
        for i in range(self.population_size):
            if eval_count >= self.budget:
                break

            solution = self._qubit_to_solution(self.qubit_population[i])
            solution = solution * (self.ub - self.lb) + self.lb  # Scale to bounds
            score = func(solution)
            eval_count += 1

            if score < self.best_score:
                self.best_score = score
                self.best_solution = solution

        return eval_count

    def _update_qubits(self):
        # Update each qubit using quantum rotation inspired by best-found solution
        best_solution_qubit = np.arccos(np.sqrt((self.best_solution - self.lb) / (self.ub - self.lb)))
        for i in range(self.population_size):
            delta_theta = self.alpha * (best_solution_qubit - self.qubit_population[i])
            self.qubit_population[i] += delta_theta
            self.qubit_population[i] = np.mod(self.qubit_population[i], 2 * np.pi)  # Keep within valid range

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population()

        eval_count = self._evaluate_population(func)
        while eval_count < self.budget:
            self._update_qubits()
            eval_count += self._evaluate_population(func)

        return self.best_solution, self.best_score
import numpy as np

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20 + self.dim
        self.alpha = 0.5  # Coefficient for updating quantum rotation gates
        self.probability_real = None
        self.probability_imaginary = None
        self.best_solution = None
        self.best_score = np.inf

    def _initialize_population(self):
        # Initialize quantum states with equal probability
        self.probability_real = np.full((self.pop_size, self.dim), 1/np.sqrt(2))
        self.probability_imaginary = np.full((self.pop_size, self.dim), 1/np.sqrt(2))

    def _measure_population(self, lb, ub):
        # Collapse quantum states to a solution vector
        collapse_real = np.random.rand(self.pop_size, self.dim) < (self.probability_real ** 2)
        collapse_imaginary = np.random.rand(self.pop_size, self.dim) < (self.probability_imaginary ** 2)
        solutions = np.where(collapse_real, lb, ub) + np.where(collapse_imaginary, 0, 1) * (ub - lb)
        return solutions

    def _update_quantum_states(self, best_solution, lb, ub):
        # Update quantum states based on the best solution
        for i in range(self.pop_size):
            for j in range(self.dim):
                delta_theta = self.alpha * ((best_solution[j] - lb[j]) / (ub[j] - lb[j]) - 0.5)
                cos_theta = np.cos(delta_theta)
                sin_theta = np.sin(delta_theta)
                real_temp = self.probability_real[i, j]
                self.probability_real[i, j] = cos_theta * self.probability_real[i, j] - sin_theta * self.probability_imaginary[i, j]
                self.probability_imaginary[i, j] = sin_theta * real_temp + cos_theta * self.probability_imaginary[i, j]

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population()

        eval_count = 0
        while eval_count < self.budget:
            solutions = self._measure_population(self.lb, self.ub)

            for sol in solutions:
                if eval_count >= self.budget:
                    break

                score = func(sol)
                eval_count += 1

                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = sol

            self._update_quantum_states(self.best_solution, self.lb, self.ub)

        return self.best_solution, self.best_score
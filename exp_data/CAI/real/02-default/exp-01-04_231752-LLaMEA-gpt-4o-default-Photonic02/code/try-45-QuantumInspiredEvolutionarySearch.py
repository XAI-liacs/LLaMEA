import numpy as np

class QuantumInspiredEvolutionarySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.q_population = np.random.rand(self.population_size, self.dim) - 0.5  # Quantum bits: range [-0.5, 0.5]
        self.solutions = np.zeros((self.population_size, self.dim))
        self.best_solution = None
        self.best_score = np.inf

    def _quantum_collapse(self, q_bit):
        return np.sign(q_bit) * np.floor(np.abs(q_bit) * 2)  # collapse to discrete {-1, 0, 1}

    def _initialize_population(self, lb, ub):
        for i in range(self.population_size):
            self.solutions[i] = (ub - lb) * (np.random.rand(self.dim)) + lb

    def _update_population(self, lb, ub):
        for i in range(self.population_size):
            q_bits = self.q_population[i] + np.random.normal(0, 0.1, self.dim)
            solution = self._quantum_collapse(q_bits)
            self.solutions[i] = (ub - lb) * (solution + 0.5) + lb
            self.solutions[i] = np.clip(self.solutions[i], lb, ub)
            self.q_population[i] = q_bits

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                score = func(self.solutions[i])
                eval_count += 1

                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = self.solutions[i]

            self._update_population(self.lb, self.ub)

        return self.best_solution, self.best_score
import numpy as np

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.q_population = np.full((self.population_size, self.dim), 0.5)  # Probabilities of qubits
        self.solutions = np.zeros((self.population_size, self.dim))
        self.best_solution = None
        self.best_score = np.inf

    def _quantum_measurement(self):
        for i in range(self.population_size):
            self.solutions[i] = np.where(np.random.rand(self.dim) < self.q_population[i], 1, 0)

    def _initialize_population(self, lb, ub):
        self.q_population = np.random.rand(self.population_size, self.dim)
        self._quantum_measurement()
        self.solutions = self.solutions * (ub - lb) + lb

    def _update_q_population(self, lb, ub):
        for i in range(self.population_size):
            candidate_solution = self.solutions[i]
            new_score = func(candidate_solution)
            if new_score < self.best_score:
                self.best_score = new_score
                self.best_solution = candidate_solution

            # Update quantum population with a simple rotation gate inspired update mechanism
            for d in range(self.dim):
                if candidate_solution[d] == self.best_solution[d]:
                    self.q_population[i][d] = self.q_population[i][d] + 0.01 * (1 - self.q_population[i][d])
                else:
                    self.q_population[i][d] = self.q_population[i][d] - 0.01 * self.q_population[i][d]

            self.q_population[i] = np.clip(self.q_population[i], 0, 1)

        self._quantum_measurement()
        self.solutions = self.solutions * (ub - lb) + lb

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                candidate_solution = self.solutions[i]
                score = func(candidate_solution)
                eval_count += 1

                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = candidate_solution

            self._update_q_population(self.lb, self.ub)

        return self.best_solution, self.best_score
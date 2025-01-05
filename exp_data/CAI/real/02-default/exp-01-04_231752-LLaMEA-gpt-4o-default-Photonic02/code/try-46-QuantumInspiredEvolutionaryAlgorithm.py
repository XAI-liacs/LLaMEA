import numpy as np

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.q_population = np.random.uniform(0, np.pi, (self.population_size, self.dim))  # Quantum population using angles
        self.best_solution = None
        self.best_score = np.inf
        self.alpha = 0.02  # Learning rate for quantum rotation

    def _qbit_rotation(self, q, best):
        # Rotate qbits towards the best solution
        delta_theta = self.alpha * (best - q)
        q_new = q + delta_theta
        q_new = np.clip(q_new, 0, np.pi)  # Keep angles within bounds
        return q_new

    def _measure_population(self, q_population, lb, ub):
        # Measure quantum population to get classical solutions
        population = np.cos(q_population) ** 2 * (ub - lb) + lb
        return population

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        eval_count = 0

        while eval_count < self.budget:
            # Measure the quantum population to get classical solutions
            population = self._measure_population(self.q_population, lb, ub)

            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                score = func(population[i])
                eval_count += 1

                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = np.copy(population[i])

            # Update the quantum population based on the best solution
            for i in range(self.population_size):
                self.q_population[i] = self._qbit_rotation(self.q_population[i], self.best_solution)

        return self.best_solution, self.best_score
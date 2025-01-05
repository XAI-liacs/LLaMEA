import numpy as np

class QuantumInspiredEvolutionarySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.alpha = 0.05  # Step size for updating
        self.quantum_population = None
        self.best_solution = None
        self.best_score = np.inf

    def _initialize_population(self, lb, ub):
        # Quantum states initialized to represent superposition of all possible states
        self.quantum_population = np.random.rand(self.population_size, self.dim, 2)
        self.quantum_population[:, :, 0] = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.quantum_population[:, :, 1] = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def _measure_population(self, lb, ub):
        # Collapse quantum states to classical solutions using probabilistic sampling
        measured_population = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            probabilities = np.random.rand(self.dim)
            measured_population[i] = np.where(probabilities < 0.5, self.quantum_population[i, :, 0], self.quantum_population[i, :, 1])
        measured_population = np.clip(measured_population, lb, ub)
        return measured_population

    def _update_quantum_states(self, population, best_solution):
        # Quantum interference and superposition update mechanism
        for i in range(self.population_size):
            for d in range(self.dim):
                theta = np.random.rand() * np.pi
                self.quantum_population[i, d, 0] += self.alpha * np.cos(theta) * (best_solution[d] - population[i, d])
                self.quantum_population[i, d, 1] += self.alpha * np.sin(theta) * (best_solution[d] - population[i, d])

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        eval_count = 0
        while eval_count < self.budget:
            population = self._measure_population(self.lb, self.ub)

            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                score = func(population[i])
                eval_count += 1

                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = population[i]

            self._update_quantum_states(population, self.best_solution)

        return self.best_solution, self.best_score
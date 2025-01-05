import numpy as np

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + self.dim
        self.alpha = 0.05  # Quantum rotation angle
        self.mutations = 0.1  # Probability of mutation
        self.q_population = None
        self.population = None
        self.best_solution = None
        self.best_score = np.inf

    def _initialize_population(self, lb, ub):
        self.q_population = np.random.rand(self.population_size, self.dim) * 2 - 1  # Q-bits initialized
        self.population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb

    def _update_q_population(self, individual, best_individual):
        # Q-bit update with rotation gate
        delta = np.arccos(np.dot(individual, best_individual) / (np.linalg.norm(individual) * np.linalg.norm(best_individual)))
        rotation_angle = self.alpha * delta
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                    [np.sin(rotation_angle), np.cos(rotation_angle)]])
        return np.dot(rotation_matrix, individual)

    def _apply_mutation(self, individual, lb, ub):
        if np.random.rand() < self.mutations:
            mutation_vector = np.random.randn(self.dim) * 0.1 * (ub - lb)
            individual += mutation_vector
            individual = np.clip(individual, lb, ub)
        return individual

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                score = func(self.population[i])
                eval_count += 1

                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = self.population[i]

            # Update quantum population based on the best solution
            for i in range(self.population_size):
                self.q_population[i] = self._update_q_population(self.q_population[i], self.best_solution)
                self.population[i] = self._apply_mutation(self.q_population[i] * (self.ub - self.lb) / 2 + (self.lb + self.ub) / 2, self.lb, self.ub)

        return self.best_solution, self.best_score
import numpy as np

class QuantumInspiredEvolutionarySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.population = None
        self.probability_amplitudes = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf
        self.alpha = 0.5  # Probability amplitude factor

    def _initialize_population(self, lb, ub):
        self.population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.probability_amplitudes = np.full((self.population_size, self.dim), 1 / np.sqrt(self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)

    def _update_population(self, lb, ub):
        for i in range(self.population_size):
            # Quantum-inspired update using probability amplitudes
            quantum_change = self.alpha * (np.random.rand(self.dim) - 0.5) * self.probability_amplitudes[i]
            mutation_vector = quantum_change

            candidate_solution = self.population[i] + mutation_vector
            candidate_solution = np.clip(candidate_solution, lb, ub)

            # Evaluate the new candidate solution
            score = func(candidate_solution)
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = candidate_solution

            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = candidate_solution

            # Update probability amplitudes for diversity
            self.probability_amplitudes[i] = (1 - self.alpha) * self.probability_amplitudes[i] + \
                                              self.alpha * np.abs(candidate_solution - self.global_best_position)

            self.population[i] = candidate_solution

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                # Evaluate current solution
                score = func(self.population[i])
                eval_count += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.population[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.population[i]

            self._update_population(self.lb, self.ub)

        return self.global_best_position, self.global_best_score
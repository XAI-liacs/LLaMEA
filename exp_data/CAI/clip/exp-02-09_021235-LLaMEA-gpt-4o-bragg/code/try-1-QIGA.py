import numpy as np

class QIGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.mutation_rate = 0.1
        self.alpha = 0.5
        self.quantum_amp = 0.5

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initialize quantum individuals
        quantum_population = np.random.uniform(-1, 1, (self.population_size, self.dim))
        real_population = np.random.uniform(lb, ub, (self.population_size, self.dim))

        personal_best_positions = np.copy(real_population)
        personal_best_scores = np.array([func(x) for x in real_population])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Quantum rotation gate
                quantum_population[i] = self.alpha * quantum_population[i] + \
                                        (1 - self.alpha) * np.random.uniform(-1, 1, self.dim)

                # Convert quantum individuals to real numbers
                real_population[i] = lb + (ub - lb) * (0.5 + 0.5 * np.sin(np.pi * quantum_population[i] + self.quantum_amp))

                # Evaluate new real position
                score = func(real_population[i])
                evaluations += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_positions[i] = real_population[i]
                    personal_best_scores[i] = score

                # Update global best
                if score < global_best_score:
                    global_best_position = real_population[i]
                    global_best_score = score

            # Mutation step
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    mutation_vector = np.random.uniform(-1, 1, self.dim)
                    quantum_population[i] = quantum_population[i] + mutation_vector
                    quantum_population[i] = np.clip(quantum_population[i], -1, 1)

        return global_best_position
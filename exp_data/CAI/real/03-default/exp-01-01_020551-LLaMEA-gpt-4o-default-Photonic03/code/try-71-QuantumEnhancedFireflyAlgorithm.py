import numpy as np

class QuantumEnhancedFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(15, dim * 2)
        self.alpha = 0.5  # Base step size
        self.beta0 = 1.0  # Base attractiveness
        self.gamma = 1.0  # Absorption coefficient
        self.quantum_step_scale = 0.1
        self.exploration_probability = 0.2  # Probability to perform a quantum jump

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        scores = np.array([func(individual) for individual in population])
        evaluations = self.population_size
        global_best_index = np.argmin(scores)
        global_best_position = population[global_best_index].copy()

        while evaluations < self.budget:
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if scores[i] > scores[j]:
                        r = np.linalg.norm(population[i] - population[j])
                        beta = self.beta0 * np.exp(-self.gamma * r**2)
                        step = self.alpha * (np.random.rand(self.dim) - 0.5) * (ub - lb)
                        movement = beta * (population[j] - population[i]) + step
                        population[i] = np.clip(population[i] + movement, lb, ub)
                        if np.random.rand() < self.exploration_probability:
                            # Perform quantum-inspired jump
                            q_step = np.random.normal(0, self.quantum_step_scale, self.dim) * (ub - lb)
                            population[i] = np.clip(global_best_position + q_step, lb, ub)

                new_score = func(population[i])
                evaluations += 1
                if new_score < scores[i]:
                    scores[i] = new_score
                    if new_score < scores[global_best_index]:
                        global_best_index = i
                        global_best_position = population[i].copy()

                if evaluations >= self.budget:
                    break

        return global_best_position, scores[global_best_index]
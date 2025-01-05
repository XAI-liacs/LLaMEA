import numpy as np

class QuantumEnhancedExplorer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(20, dim)
        self.alpha = 0.8  # Initial exploration factor for search space contraction
        self.beta = 0.05  # Quantum-inspired adjustment rate
        self.sigma_decay = 0.99  # Decay rate for quantum adjustment range

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        scores = np.array([func(ind) for ind in population])
        best_index = np.argmin(scores)
        best_position = population[best_index].copy()
        evaluations = self.population_size
        sigma = (ub - lb) * 0.1  # Initial quantum adjustment range

        while evaluations < self.budget:
            # Adaptive contraction of search space
            contraction_center = (1 - self.alpha) * best_position + self.alpha * np.mean(population, axis=0)
            new_population = np.random.uniform(
                np.maximum(lb, contraction_center - (ub - lb) / 2 * self.alpha),
                np.minimum(ub, contraction_center + (ub - lb) / 2 * self.alpha),
                (self.population_size, self.dim)
            )

            # Quantum-inspired exploration
            for i in range(self.population_size):
                if np.random.rand() < self.beta:
                    q = np.random.normal(0, sigma, self.dim)
                    new_population[i] = np.clip(best_position + q, lb, ub)

            new_scores = np.array([func(ind) for ind in new_population])
            evaluations += self.population_size

            # Update best position
            for i in range(self.population_size):
                if new_scores[i] < scores[best_index]:
                    best_index = i
                    best_position = new_population[best_index].copy()

            population = new_population
            scores = new_scores
            sigma *= self.sigma_decay  # Decay the quantum adjustment range

        return best_position, scores[best_index]
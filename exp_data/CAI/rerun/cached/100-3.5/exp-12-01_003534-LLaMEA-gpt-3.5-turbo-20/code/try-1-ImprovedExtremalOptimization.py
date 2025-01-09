import numpy as np

class ImprovedExtremalOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_step = 0.5

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        evaluations = np.apply_along_axis(func, 1, population)
        for _ in range(self.budget):
            max_idx = np.argmax(evaluations)
            candidate = population[max_idx] + np.random.normal(0, self.mutation_step, self.dim)
            candidate = np.clip(candidate, -5.0, 5.0)
            candidate_eval = func(candidate)
            if candidate_eval < evaluations[max_idx]:
                population[max_idx] = candidate
                evaluations[max_idx] = candidate_eval
                self.mutation_step *= 0.99  # Adaptive mutation step update
            else:
                self.mutation_step *= 1.01
        best_idx = np.argmin(evaluations)
        return population[best_idx]
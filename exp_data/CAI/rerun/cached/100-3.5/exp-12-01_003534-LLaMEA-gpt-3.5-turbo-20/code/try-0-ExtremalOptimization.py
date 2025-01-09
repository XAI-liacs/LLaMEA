import numpy as np

class ExtremalOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        evaluations = np.apply_along_axis(func, 1, population)
        for _ in range(self.budget):
            max_idx = np.argmax(evaluations)
            candidate = np.random.uniform(-5.0, 5.0, self.dim)
            candidate_eval = func(candidate)
            if candidate_eval < evaluations[max_idx]:
                population[max_idx] = candidate
                evaluations[max_idx] = candidate_eval
        best_idx = np.argmin(evaluations)
        return population[best_idx]
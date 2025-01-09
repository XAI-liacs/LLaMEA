from scipy.optimize import differential_evolution

class NovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(-5.0, 5.0)] * self.dim
        result = differential_evolution(func, bounds, maxiter=self.budget)
        return result.x
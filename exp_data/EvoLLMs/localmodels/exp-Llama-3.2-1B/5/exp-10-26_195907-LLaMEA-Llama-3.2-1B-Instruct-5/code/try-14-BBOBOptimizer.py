import random
import numpy as np
from scipy.optimize import minimize

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        while True:
            for _ in range(self.budget):
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                if np.linalg.norm(func(x)) < self.budget / 2:
                    return x
            x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            self.search_space = np.vstack((self.search_space, x))
            self.search_space = np.delete(self.search_space, 0, axis=0)

    def _evaluate_bbob(self, func, budget):
        return minimize(lambda x: func(x), np.zeros(self.budget), args=(x, self.budget), method="SLSQP", bounds=self.search_space, options={"maxiter": 1000})

    def optimize(self, func):
        self.func = func
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(self.dim, 2))
        self.budget = 1000
        return self._evaluate_bbob(self.func, self.budget)
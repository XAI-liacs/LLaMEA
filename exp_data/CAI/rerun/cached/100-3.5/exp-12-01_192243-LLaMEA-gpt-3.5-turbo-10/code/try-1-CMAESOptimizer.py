import numpy as np
import cma

class CMAESOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        es = cma.CMAEvolutionStrategy(np.zeros(self.dim), 0.5).optimize(func, iterations=self.budget)
        return es.result.xbest
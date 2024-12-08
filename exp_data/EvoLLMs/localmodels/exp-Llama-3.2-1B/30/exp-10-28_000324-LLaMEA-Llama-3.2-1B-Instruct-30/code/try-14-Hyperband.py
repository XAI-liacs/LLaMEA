import numpy as np
import random

class Hyperband:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.explore_count = 0
        self.best_func = None
        self.best_func_val = None
        self.func_evals = []
        self.budget_count = 0
        self.reduction_factor = 0.3
        self.tolerance = 1e-6

    def __call__(self, func):
        while self.explore_count < self.budget:
            if self.explore_count > 100:  
                break
            func_eval = func(np.random.uniform(-5.0, 5.0, self.dim))
            self.func_evals.append(func_eval)
            if self.best_func is None or np.abs(func_eval - self.best_func_val) > np.abs(func_eval - self.best_func):
                self.best_func = func_eval
                self.best_func_val = func_eval
            self.explore_count += 1
            self.budget_count += 1
            if self.budget_count > self.budget / 2:
                break
            if random.random() < self.explore_rate:
                self.explore_count -= 1
                if random.random() < self.reduction_factor:
                    self.explore_count += 1
            if np.abs(func_eval - self.best_func_val) < self.tolerance:
                break
        return self.best_func

# One-line description: Hyperband is a novel metaheuristic that uses hyperband search to efficiently explore the search space of black box functions.
# Code: 
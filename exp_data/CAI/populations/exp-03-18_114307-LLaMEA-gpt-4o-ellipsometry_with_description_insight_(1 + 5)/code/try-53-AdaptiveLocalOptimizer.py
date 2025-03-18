import numpy as np
from scipy.optimize import minimize, differential_evolution

class AdaptiveLocalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')

        # Utilize differential evolution for initial exploration
        init_result = differential_evolution(self.bounded_func(func, lb, ub),
                                             bounds=list(zip(lb, ub)), strategy='best1bin',
                                             maxiter=int(self.budget * 0.3 / (self.dim + 1)),
                                             polish=False, disp=False)
        self.evaluations += init_result.nfev

        if init_result.fun < best_value:
            best_value = init_result.fun
            best_solution = init_result.x

        # Refine using local optimizer starting from the best DE solution
        if self.evaluations < self.budget:
            result = minimize(self.bounded_func(func, lb, ub), best_solution,
                              method='Nelder-Mead',
                              options={'maxfev': self.budget - self.evaluations})
            self.evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution

    def bounded_func(self, func, lb, ub):
        def func_with_bounds(x):
            x_clipped = np.clip(x, lb, ub)
            return func(x_clipped)
        return func_with_bounds
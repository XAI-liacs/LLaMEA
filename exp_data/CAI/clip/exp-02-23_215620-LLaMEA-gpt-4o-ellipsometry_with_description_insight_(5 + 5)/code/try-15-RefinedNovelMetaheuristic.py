import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class RefinedNovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        sampler = Sobol(d=self.dim, scramble=True)
        initial_guesses = sampler.random(n=min(2**int(np.log2(self.budget/self.dim)), self.budget//2))
        initial_guesses = lb + (ub - lb) * initial_guesses

        best_solution = None
        best_objective = float('inf')

        def track_evaluations(x):
            if self.evaluations < self.budget:
                self.evaluations += 1
                return func(x)
            else:
                raise Exception("Exceeded budget of function evaluations")

        for guess in initial_guesses:
            if self.evaluations >= self.budget:
                break

            result = minimize(track_evaluations, guess, method='BFGS', bounds=list(zip(lb, ub)))
            
            if result.fun < best_objective:
                best_solution = result.x
                best_objective = result.fun

        if self.evaluations < self.budget:
            refined_bounds = [(max(lb[i], best_solution[i] - (ub[i] - lb[i]) * 0.1), 
                               min(ub[i], best_solution[i] + (ub[i] - lb[i]) * 0.1)) 
                              for i in range(self.dim)]

            result = minimize(track_evaluations, best_solution, method='BFGS', bounds=refined_bounds)
            if result.fun < best_objective:
                best_solution = result.x
                best_objective = result.fun

        return best_solution
import numpy as np
from scipy.optimize import minimize

class AdaptiveGradientBoostedSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Retrieve bounds
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        # Step 1: Initial Exploration with Adaptive Sampling
        initial_samples = min(self.budget // 3 + 1, 20 * self.dim)
        samples = np.random.uniform(low=lb, high=ub, size=(initial_samples, self.dim))
        evaluations = []
        
        for s in samples:
            if self.evaluations >= self.budget:
                break
            eval_result = func(s)
            evaluations.append((eval_result, s))
            self.evaluations += 1
        
        # Find the best sample from initial exploration
        evaluations.sort(key=lambda x: x[0])
        best_sample = evaluations[0][1]
        
        # Step 2: Adaptive Gradient-Boosted Local Search
        if self.evaluations < self.budget:
            local_budget = self.budget - self.evaluations
            step_size = (ub - lb) / 20.0  # Adaptive step size
            for _ in range(local_budget):
                grad = self.approximate_gradient(func, best_sample, step_size)
                step_direction = -grad / np.linalg.norm(grad)
                next_sample = np.clip(best_sample + step_direction * step_size, lb, ub)
                eval_result = func(next_sample)
                self.evaluations += 1
                if eval_result < evaluations[0][0]:
                    evaluations[0] = (eval_result, next_sample)
                    best_sample = next_sample
        
        return best_sample

    def approximate_gradient(self, func, x, step_size):
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_step = np.copy(x)
            x_step[i] += step_size[i]
            gradient[i] = (func(x_step) - func(x)) / step_size[i]
        return gradient
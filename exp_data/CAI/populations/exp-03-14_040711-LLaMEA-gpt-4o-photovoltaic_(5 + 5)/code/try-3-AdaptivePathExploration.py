import numpy as np

class AdaptivePathExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = np.random.uniform(lb, ub, self.dim)
        best_value = func(best_solution)
        evaluations = 1

        while evaluations < self.budget:
            alpha = 1.0 - (evaluations / self.budget)  # Dynamic adjustment factor
            gradient = self.estimate_gradient(func, best_solution, lb, ub)
            candidate_solution = best_solution + alpha * gradient + np.random.uniform(-0.1, 0.1, self.dim)
            candidate_solution = np.clip(candidate_solution, lb, ub)
            candidate_value = func(candidate_solution)
            evaluations += 1
            
            if candidate_value < best_value:
                best_solution, best_value = candidate_solution, candidate_value
        
        return best_solution

    def estimate_gradient(self, func, solution, lb, ub):
        epsilon = 1e-8
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            step = np.zeros(self.dim)
            step[i] = epsilon
            upper_value = func(np.clip(solution + step, lb, ub))
            lower_value = func(np.clip(solution - step, lb, ub))
            grad[i] = (upper_value - lower_value) / (2 * epsilon)
        return grad
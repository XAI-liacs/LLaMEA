import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class GradientEnhancedHybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Define bounds and initialize Sobol sequence
        lb, ub = func.bounds.lb, func.bounds.ub
        sobol = Sobol(d=self.dim, scramble=True)
        
        # Initial sampling with Sobol sequence
        n_init = min(self.budget // 3, 50)  # Use a third of the budget or 50 samples
        samples = sobol.random_base2(m=int(np.log2(n_init)))
        scaled_samples = lb + samples * (ub - lb)

        # Evaluate initial samples
        best_x = None
        best_f = float('inf')
        evaluations = 0

        for x in scaled_samples:
            f_val = func(x)
            evaluations += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x

        # Define a gradient approximation function
        def approx_gradient(x):
            epsilon = 1e-5
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = np.array(x, copy=True)
                x_plus[i] += epsilon
                grad[i] = (func(x_plus) - func(x)) / epsilon
            return grad

        # Local optimization with BFGS using gradient approximation
        def objective(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                raise Exception("Budget exceeded")
            evaluations += 1
            return func(x)

        result = minimize(objective, best_x, method='L-BFGS-B', jac=approx_gradient, bounds=[(l, u) for l, u in zip(lb, ub)])

        return result.x

# Usage example:
# optimizer = GradientEnhancedHybrid(budget=100, dim=2)
# best_parameters = optimizer(my_black_box_function)
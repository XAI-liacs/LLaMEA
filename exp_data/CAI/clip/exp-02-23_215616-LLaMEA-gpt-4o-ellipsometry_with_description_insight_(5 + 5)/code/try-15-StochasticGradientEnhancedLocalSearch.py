import numpy as np
from scipy.optimize import minimize

class StochasticGradientEnhancedLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds for the search space
        lower_bounds = np.array(func.bounds.lb)
        upper_bounds = np.array(func.bounds.ub)

        # Initial uniform random sampling
        num_initial_samples = min(self.budget // 5, 10)
        samples = np.random.uniform(lower_bounds, upper_bounds, size=(num_initial_samples, self.dim))

        # Evaluate the initial samples
        evaluations = []
        for sample in samples:
            if len(evaluations) < self.budget:
                evaluations.append((sample, func(sample)))
            else:
                break

        # Sort initial samples based on their function value
        evaluations.sort(key=lambda x: x[1])
        best_sample, best_value = evaluations[0]

        # Stochastic gradient estimation using finite differences
        def stochastic_gradient(x):
            epsilon = 1e-5
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_upper = np.copy(x)
                x_upper[i] += epsilon
                f_upper = func(x_upper)
                grad[i] = (f_upper - best_value) / epsilon
                if len(evaluations) < self.budget:
                    evaluations.append((x_upper, f_upper))
                else:
                    break
            return grad

        # Define a local optimization function using BFGS with stochastic gradients
        def local_optimization(x0):
            grad = stochastic_gradient(x0)
            res = minimize(lambda x: func(x), x0, method='L-BFGS-B', jac=lambda x: grad,
                           bounds=list(zip(lower_bounds, upper_bounds)),
                           options={'maxiter': self.budget - len(evaluations)})
            return res.x, res.fun

        # Conduct local optimization from the best initial sample
        if len(evaluations) < self.budget:
            solution, value = local_optimization(best_sample)
            if value < best_value:
                best_sample, best_value = solution, value

        # Adjust bounds dynamically based on best found solution
        margin = 0.1 + 0.05 * np.random.rand()
        new_lower_bounds = np.maximum(lower_bounds, best_sample - margin * best_sample)
        new_upper_bounds = np.minimum(upper_bounds, best_sample + margin * best_sample)

        # Final local optimization with adjusted bounds
        if len(evaluations) < self.budget:
            final_solution, final_value = local_optimization(best_sample)
            if final_value < best_value:
                best_sample, best_value = final_solution, final_value

        return best_sample, best_value
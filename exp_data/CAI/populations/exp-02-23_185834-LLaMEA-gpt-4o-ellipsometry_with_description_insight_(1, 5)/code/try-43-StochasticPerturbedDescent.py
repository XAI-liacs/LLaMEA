import numpy as np
from scipy.optimize import minimize

class StochasticPerturbedDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        # Step 1: Uniform sampling with stochastic perturbations
        initial_samples = min(self.budget // 3, 40 * self.dim)
        samples = np.random.uniform(low=lb, high=ub, size=(initial_samples, self.dim))
        evaluations = []
        
        for s in samples:
            if self.evaluations >= self.budget:
                break
            perturbed_sample = s + np.random.normal(0, 0.1, size=self.dim)
            perturbed_sample = np.clip(perturbed_sample, lb, ub)
            eval_result = func(perturbed_sample)
            evaluations.append((eval_result, perturbed_sample))
            self.evaluations += 1
        
        evaluations.sort(key=lambda x: x[0])
        top_samples = [e[1] for e in evaluations[:10]]

        # Step 2: Local optimization with perturbed starting points
        best_sample = None
        best_value = float('inf')
        for sample in top_samples:
            if self.evaluations < self.budget:
                local_budget = self.budget - self.evaluations
                options = {'maxiter': local_budget, 'gtol': 1e-8}
                perturbed_sample = sample + np.random.normal(0, 0.05, size=self.dim)
                perturbed_sample = np.clip(perturbed_sample, lb, ub)
                result = minimize(func, perturbed_sample, method='BFGS', bounds=list(zip(lb, ub)), options=options)
                if result.success and result.fun < best_value:
                    best_sample = result.x
                    best_value = result.fun
                    if best_value < 1e-6:
                        break
                self.evaluations += result.nfev
        
        return best_sample if best_sample is not None else top_samples[0]
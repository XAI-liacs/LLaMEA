import numpy as np
from scipy.optimize import minimize

class AWSRO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Retrieve bounds
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        # Step 1: Adaptive Weighted Sampling
        initial_samples = min(int(self.budget // 2), 10 * self.dim)
        samples = np.random.uniform(low=lb, high=ub, size=(initial_samples, self.dim))
        evaluations = []
        
        for s in samples:
            if self.evaluations >= self.budget:
                break
            eval_result = func(s)
            evaluations.append((eval_result, s))
            self.evaluations += 1
        
        # Sort samples by evaluation
        evaluations.sort(key=lambda x: x[0])
        
        # Assign weights inversely proportional to the cost
        weights = np.array([1 / (e[0] + 1e-9) for e in evaluations])
        weights /= weights.sum()
        
        # Weighted choice of samples for local search
        selected_indices = np.random.choice(len(evaluations), size=min(3, len(evaluations)), p=weights)
        
        # Step 2: Refined Local Optimization for top weighted samples
        best_eval = evaluations[0][0]
        best_sample = evaluations[0][1]
        for idx in selected_indices:
            if self.evaluations >= self.budget:
                break
            local_budget = (self.budget - self.evaluations) // len(selected_indices)
            if local_budget < 1:
                break
            options = {'maxiter': local_budget}
            result = minimize(func, evaluations[idx][1], method='L-BFGS-B', bounds=list(zip(lb, ub)), options=options)
            self.evaluations += result.nfev
            if result.success and result.fun < best_eval:
                best_eval = result.fun
                best_sample = result.x
        
        return best_sample
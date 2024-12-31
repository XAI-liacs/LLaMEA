import numpy as np

class EvolutionaryStrategy:
    def __init__(self, budget=10000, dim=10, lambda_=20, sigma=0.1):
        self.budget = budget
        self.dim = dim
        self.lambda_ = lambda_
        self.sigma = sigma
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        x_mean = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        
        for i in range(self.budget // self.lambda_):
            offspring = np.random.normal(0, self.sigma, (self.lambda_, self.dim)) + x_mean
            f_vals = [func(x) for x in offspring]
            
            min_idx = np.argmin(f_vals)
            if f_vals[min_idx] < self.f_opt:
                self.f_opt = f_vals[min_idx]
                self.x_opt = offspring[min_idx]
            
            x_mean = np.mean(offspring, axis=0)
        
        return self.f_opt, self.x_opt
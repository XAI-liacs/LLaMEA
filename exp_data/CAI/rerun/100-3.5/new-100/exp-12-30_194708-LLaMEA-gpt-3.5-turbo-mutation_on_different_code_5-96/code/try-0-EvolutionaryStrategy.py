import numpy as np

class EvolutionaryStrategy:
    def __init__(self, budget=10000, dim=10, mu=10, lambda_=100, tau=1/np.sqrt(2*np.sqrt(dim)), tau_prime=1/np.sqrt(2*dim)):
        self.budget = budget
        self.dim = dim
        self.mu = mu
        self.lambda_ = lambda_
        self.tau = tau
        self.tau_prime = tau_prime
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        x_mean = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        sigma = np.ones(self.dim)
        
        for _ in range(self.budget // self.lambda_):
            offspring = np.random.normal(0, 1, (self.lambda_, self.dim))
            population = np.tile(x_mean, (self.lambda_, 1)) + sigma * offspring
            
            f_vals = [func(x) for x in population]
            
            idx_best = np.argmin(f_vals)
            if f_vals[idx_best] < self.f_opt:
                self.f_opt = f_vals[idx_best]
                self.x_opt = population[idx_best]
            
            sorted_indices = np.argsort(f_vals)
            selected = population[sorted_indices[:self.mu]]
            
            x_mean = np.mean(selected, axis=0)
            sigma *= np.exp(self.tau_prime*np.random.normal(0, 1, self.dim) + self.tau*np.random.normal(0, 1))
        
        return self.f_opt, self.x_opt
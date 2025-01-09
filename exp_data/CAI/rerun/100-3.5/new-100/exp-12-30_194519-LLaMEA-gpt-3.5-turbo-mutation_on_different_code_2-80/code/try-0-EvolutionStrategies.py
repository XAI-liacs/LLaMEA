import numpy as np

class EvolutionStrategies:
    def __init__(self, budget=10000, dim=10, mu=10, lambda_=100, sigma_init=1.0):
        self.budget = budget
        self.dim = dim
        self.mu = mu
        self.lambda_ = lambda_
        self.sigma = sigma_init
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        C = np.eye(self.dim)
        x_mean = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        
        for i in range(self.budget // self.lambda_):
            offspring = np.random.multivariate_normal(np.zeros(self.dim), C, self.lambda_)
            x_offspring = np.tile(x_mean, (self.lambda_, 1)) + self.sigma * offspring
            f_values = np.array([func(x) for x in x_offspring])
            
            best_idx = np.argmin(f_values)
            if f_values[best_idx] < self.f_opt:
                self.f_opt = f_values[best_idx]
                self.x_opt = x_offspring[best_idx]
                
            selected_offspring = offspring[best_idx]
            x_mean += (1 / self.mu) * self.sigma * np.dot(selected_offspring, offspring)
            C = (1 - 1/self.mu) * C + (1/self.mu) * np.outer(selected_offspring, selected_offspring)

        return self.f_opt, self.x_opt
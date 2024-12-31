class EvolutionaryStrategy:
    def __init__(self, budget=10000, dim=10, mu=10, lambda_=100, sigma=0.1, adapt_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.mu = mu
        self.lambda_ = lambda_
        self.sigma = sigma
        self.adapt_rate = adapt_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.lambda_, self.dim))
        
        for i in range(self.budget // self.lambda_):
            self.sigma *= np.exp(self.adapt_rate * np.random.normal(0, 1))
            offspring = population + np.random.normal(0, self.sigma, size=(self.lambda_, self.dim))
            values = [func(x) for x in offspring]
            
            selected_indices = np.argsort(values)[:self.mu]
            population = offspring[selected_indices]
            
            f_min = np.min(values)
            if f_min < self.f_opt:
                self.f_opt = f_min
                self.x_opt = population[np.argmin(values)]
        
        return self.f_opt, self.x_opt
import numpy as np

class Improved_EMO_DE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9, strategy_adaptation=True):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.strategy_adaptation = strategy_adaptation
        self.f_opt = np.Inf
        self.x_opt = None
        self.strategy_params = {'F_min': 0.2, 'F_max': 0.8, 'CR_min': 0.1, 'CR_max': 0.9}
        
    def adapt_parameters(self, iteration):
        if self.strategy_adaptation:
            self.F = self.strategy_params['F_min'] + (self.strategy_params['F_max'] - self.strategy_params['F_min']) * (iteration / self.budget)
            self.CR = self.strategy_params['CR_max'] - (self.strategy_params['CR_max'] - self.strategy_params['CR_min']) * (iteration / self.budget)
        
    def __call__(self, func):
        def mutate(x_r1, x_r2, x_r3):
            return x_r1 + self.F * (x_r2 - x_r3)

        def crossover(x, v):
            j_rand = np.random.randint(self.dim)
            u = np.array([v[i] if i == j_rand or np.random.rand() < self.CR else x[i] for i in range(self.dim)])
            return u

        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))

        for i in range(self.budget // self.pop_size):
            self.adapt_parameters(i)
            for j in range(self.pop_size):
                x = population[j]
                idx_r1, idx_r2, idx_r3 = np.random.choice(np.delete(np.arange(self.pop_size), j), 3, replace=False)
                x_r1, x_r2, x_r3 = population[idx_r1], population[idx_r2], population[idx_r3]
                
                v = mutate(x_r1, x_r2, x_r3)
                u = crossover(x, v)
                
                f_u = func(u)
                if f_u < self.f_opt:
                    self.f_opt = f_u
                    self.x_opt = u
                population[j] = u if f_u < func(x) else x

        return self.f_opt, self.x_opt
import numpy as np

class DynamicCuckooSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, 1, size)
        v = np.random.normal(0, 1, size)
        step = u / (abs(v) ** (1 / beta))
        return 0.01 * step
        
    def __call__(self, func):
        n = self.budget
        Nc = 5
        pa = 0.25
        x_best = np.random.uniform(-5, 5, self.dim)
        f_best = func(x_best)
        
        for i in range(n):
            x_new = x_best + self.levy_flight(self.dim)
            x_new = np.clip(x_new, -5, 5)
            f_new = func(x_new)
            
            if f_new < f_best:
                x_best = x_new
                f_best = f_new
            
            if np.random.rand() < pa:
                x_cuckoo = np.random.uniform(-5, 5, self.dim)
                f_cuckoo = func(x_cuckoo)
                if f_cuckoo < f_best:
                    x_best = x_cuckoo
                    f_best = f_cuckoo
            
            if i % Nc == 0:
                pa = max(0.1, pa * 0.95)
        
        return x_best
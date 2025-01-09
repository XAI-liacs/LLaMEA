import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        def pso(x, v, pbest, gbest, c1, c2):
            for i in range(len(x)):
                r1, r2 = np.random.random(), np.random.random()
                v[i] = 0.5 * v[i] + c1 * r1 * (pbest[i] - x[i]) + c2 * r2 * (gbest[i] - x[i])
                x[i] = np.clip(x[i] + v[i], -5.0, 5.0)
            return x, v
        
        def sa(x, best_x, T, alpha):
            new_x = x + np.random.uniform(-1, 1, len(x)) * T
            new_x = np.clip(new_x, -5.0, 5.0)
            old_cost, new_cost = func(x), func(new_x)
            if new_cost < old_cost or np.random.rand() < np.exp((old_cost - new_cost) / T):
                return new_x if new_cost < func(best_x) else best_x
            return x
        
        x = np.random.uniform(-5.0, 5.0, self.dim)
        v = np.zeros(self.dim)
        pbest = x.copy()
        gbest = x.copy()
        
        for _ in range(self.budget):
            c1, c2 = 2.0, 2.0
            x, v = pso(x, v, pbest, gbest, c1, c2)
            pbest = x if func(x) < func(pbest) else pbest
            gbest = x if func(x) < func(gbest) else gbest
            T = 5.0 * (1 - _ / self.budget)
            x = sa(x, gbest, T, 0.95)
        
        return gbest
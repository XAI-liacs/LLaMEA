import numpy as np

class HFADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 30
        alpha = 0.1
        beta_min = 0.2
        beta_max = 1.0
        scale = 0.01
        
        def levy_flight():
            beta = beta_min + (beta_max - beta_min) * np.random.rand()
            b = np.random.normal(0.0, scale, self.dim)
            l = np.random.normal(0.0, 1.0, self.dim)
            levy = b / np.power(np.abs(l), 1.0 / beta)
            return levy
        
        def de(x, a, b, c):
            return np.clip(a + beta_min * (b - c), -5.0, 5.0)
        
        def firefly_swarm():
            swarm = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
            best_point = swarm[np.argmin([func(p) for p in swarm])]
            
            for _ in range(self.budget):
                for i in range(pop_size):
                    for j in range(pop_size):
                        if func(swarm[j]) < func(swarm[i]):
                            step = alpha * levy_flight()
                            swarm[i] += step * (swarm[j] - swarm[i])
                            swarm[i] = np.array([de(swarm[i, d], swarm[i, d], best_point[d], swarm[j, d]) for d in range(self.dim)])
                
            return best_point

        return firefly_swarm()
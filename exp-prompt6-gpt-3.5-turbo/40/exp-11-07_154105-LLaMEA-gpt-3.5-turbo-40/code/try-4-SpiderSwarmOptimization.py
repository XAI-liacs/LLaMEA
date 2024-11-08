import numpy as np

class SpiderSwarmOptimization:
    def __init__(self, budget, dim, swarm_size=10, alpha=0.1, beta=2.0):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.alpha = alpha
        self.beta = beta

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        costs = np.array([func(x) for x in swarm])
        
        for _ in range(self.budget // self.swarm_size):
            for i in range(self.swarm_size):
                new_pos = swarm[i] + self.alpha * np.mean(swarm, axis=0) + self.beta * np.random.uniform(-1, 1, self.dim)
                new_pos = np.clip(new_pos, -5.0, 5.0)
                new_cost = func(new_pos)
                
                if new_cost < costs[i]:
                    swarm[i] = new_pos
                    costs[i] = new_cost
        
        best_idx = np.argmin(costs)
        best_solution = swarm[best_idx]
        return best_solution
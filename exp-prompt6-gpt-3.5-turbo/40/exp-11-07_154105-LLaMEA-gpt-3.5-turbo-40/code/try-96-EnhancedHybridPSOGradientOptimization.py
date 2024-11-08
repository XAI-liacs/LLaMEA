import numpy as np

class EnhancedHybridPSOGradientOptimization:
    def __init__(self, budget, dim, swarm_size=10, alpha=0.1, beta=2.0, learning_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.alpha = alpha
        self.beta = beta
        self.learning_rate = learning_rate

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        costs = np.array([func(x) for x in swarm])
        
        for _ in range(self.budget // self.swarm_size):
            gradients = np.mean(swarm, axis=0) - swarm
            perturbation = self.alpha * gradients + self.beta * np.random.uniform(-1, 1, (self.swarm_size, self.dim))
            new_pos = np.clip(swarm + perturbation, -5.0, 5.0)
            new_costs = np.array([func(x) for x in new_pos])
            
            improved_indices = new_costs < costs
            swarm[improved_indices] = new_pos[improved_indices]
            costs = np.where(improved_indices, new_costs, costs)  # Only update costs where improvements occurred
        
        best_idx = np.argmin(costs)
        best_solution = swarm[best_idx]
        return best_solution
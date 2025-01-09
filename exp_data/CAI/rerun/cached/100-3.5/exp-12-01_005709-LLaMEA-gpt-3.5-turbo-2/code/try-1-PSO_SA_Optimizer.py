import numpy as np

class PSO_SA_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, max_iter=100):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.max_iter = max_iter
    
    def __call__(self, func):
        def objective_function(x):
            return func(x)
        
        def simulated_annealing(current, new, temp):
            cost_diff = objective_function(new) - objective_function(current)
            if cost_diff < 0 or np.random.rand() < np.exp(-cost_diff / temp):
                return new
            return current
        
        swarm = np.random.uniform(low=-5.0, high=5.0, size=(self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        pbest = swarm.copy()
        gbest = pbest[np.argmin([objective_function(p) for p in pbest])]
        best_cost = objective_function(gbest)
        
        temperature = 1.0
        cooling_rate = 0.95
        
        for _ in range(self.max_iter):
            for i in range(self.swarm_size):
                new_vel = 0.5 * velocities[i] + 2.0 * np.random.rand() * (pbest[i] - swarm[i]) + 2.0 * np.random.rand() * (gbest - swarm[i])
                swarm[i] = swarm[i] + new_vel
                swarm[i] = np.clip(swarm[i], -5.0, 5.0)
                pbest[i] = simulated_annealing(pbest[i], swarm[i], temperature)
                
                if objective_function(pbest[i]) < objective_function(gbest):
                    gbest = pbest[i]
            
            temperature *= cooling_rate
        
        return gbest
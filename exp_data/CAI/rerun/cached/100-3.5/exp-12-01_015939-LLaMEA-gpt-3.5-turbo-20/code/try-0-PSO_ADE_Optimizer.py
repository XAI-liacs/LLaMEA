import numpy as np

class PSO_ADE_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, pso_iter=100, de_iter=100):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.pso_iter = pso_iter
        self.de_iter = de_iter

    def __call__(self, func):
        def fitness(x):
            return func(x)
        
        def within_bounds(x):
            return np.clip(x, -5.0, 5.0)
        
        def initialize_swarm():
            return np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))

        swarm = initialize_swarm()
        best_position = swarm[np.argmin([fitness(p) for p in swarm])]
        
        for _ in range(self.pso_iter):
            for i in range(self.swarm_size):
                new_position = swarm[i] + np.random.uniform() * (best_position - swarm[i])
                new_position = within_bounds(new_position)
                
                if fitness(new_position) < fitness(swarm[i]):
                    swarm[i] = new_position
                    if fitness(new_position) < fitness(best_position):
                        best_position = new_position
        
        for _ in range(self.de_iter):
            for i in range(self.swarm_size):
                mutant = np.clip(swarm[np.random.choice(self.swarm_size)] + np.random.uniform(-1, 1) * (swarm[np.random.choice(self.swarm_size)] - swarm[np.random.choice(self.swarm_size)]), -5.0, 5.0)
                trial = swarm[i] + np.random.uniform() * (mutant - swarm[i])
                trial = within_bounds(trial)
                
                if fitness(trial) < fitness(swarm[i]):
                    swarm[i] = trial
                    if fitness(trial) < fitness(best_position):
                        best_position = trial

        return best_position
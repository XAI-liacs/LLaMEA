import numpy as np

class PSO_SA_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, max_local_evals=50, max_iter=100):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.max_local_evals = max_local_evals
        self.max_iter = max_iter

    def __call__(self, func):
        swarm_position = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        swarm_velocity = np.zeros((self.swarm_size, self.dim))
        personal_best = swarm_position.copy()
        global_best = personal_best[np.argmin([func(p) for p in personal_best])]
        
        for _ in range(self.max_iter):
            for i in range(self.swarm_size):
                # Update velocity
                inertia_weight = 0.5 + 0.5 * np.cos((np.pi * _)/self.max_iter)
                cognitive_component = 1.5 * np.random.rand(self.dim) * (personal_best[i] - swarm_position[i])
                social_component = 1.5 * np.random.rand(self.dim) * (global_best - swarm_position[i])
                swarm_velocity[i] = inertia_weight * swarm_velocity[i] + cognitive_component + social_component
                
                # Update position
                swarm_position[i] = np.clip(swarm_position[i] + swarm_velocity[i], -5.0, 5.0)
                
                # Local search with Simulated Annealing
                local_position = swarm_position[i].copy()
                local_best = local_position.copy()
                for _ in range(self.max_local_evals):
                    new_position = local_position + np.random.normal(0, 0.1, self.dim)
                    if func(new_position) < func(local_position):
                        local_position = new_position
                        if func(local_position) < func(local_best):
                            local_best = local_position
                    else:
                        delta_E = func(new_position) - func(local_position)
                        if np.random.rand() < np.exp(-delta_E):
                            local_position = new_position
                
                swarm_position[i] = local_best
                
                # Update personal best and global best
                if func(swarm_position[i]) < func(personal_best[i]):
                    personal_best[i] = swarm_position[i]
                if func(personal_best[i]) < func(global_best):
                    global_best = personal_best[i]
        
        return global_best
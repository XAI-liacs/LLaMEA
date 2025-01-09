import numpy as np

class HybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        def pso_sa_optimizer():
            # PSO initialization
            swarm_size = 20
            max_iter = 100
            inertia_weight = 0.5
            cognitive_weight = 1.5
            social_weight = 1.5
            p_best_pos = np.random.uniform(-5.0, 5.0, (swarm_size, self.dim))
            p_best_val = np.full((swarm_size,), np.inf)
            g_best_pos = np.zeros(self.dim)
            g_best_val = np.inf
            velocities = np.zeros((swarm_size, self.dim))
            
            # SA initialization
            initial_temp = 100.0
            final_temp = 0.1
            cooling_rate = 0.95
            curr_temp = initial_temp
            curr_pos = np.random.uniform(-5.0, 5.0, self.dim)
            best_pos = curr_pos
            best_val = np.inf
            
            for _ in range(max_iter):
                for i in range(swarm_size):
                    # PSO update
                    velocities[i] = inertia_weight * velocities[i] + cognitive_weight * np.random.rand() * (p_best_pos[i] - curr_pos) + social_weight * np.random.rand() * (g_best_pos - curr_pos)
                    curr_pos = curr_pos + velocities[i]
                    
                    # SA update
                    new_pos = curr_pos + np.random.normal(0, 0.1, self.dim)
                    if func(new_pos) < func(curr_pos) or np.random.rand() < np.exp((func(curr_pos) - func(new_pos)) / curr_temp):
                        curr_pos = new_pos
                    curr_temp = curr_temp * cooling_rate
                    
                    # Update PSO best
                    curr_val = func(curr_pos)
                    if curr_val < p_best_val[i]:
                        p_best_val[i] = curr_val
                        p_best_pos[i] = curr_pos
                    if curr_val < best_val:
                        best_val = curr_val
                        best_pos = curr_pos
                if best_val < g_best_val:
                    g_best_val = best_val
                    g_best_pos = best_pos
                
            return g_best_pos
        
        return pso_sa_optimizer()
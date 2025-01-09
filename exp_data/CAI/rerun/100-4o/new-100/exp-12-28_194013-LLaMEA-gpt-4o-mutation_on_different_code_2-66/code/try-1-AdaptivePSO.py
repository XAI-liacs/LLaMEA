import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5.0, 5.0)
        
    def __call__(self, func):
        lb, ub = self.bounds
        # Initialize the swarm
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.swarm_size, np.inf)
        global_best_position = None
        global_best_score = np.inf
        
        # PSO parameters
        inertia_weight = 0.7
        cognitive_coeff = 1.5
        social_coeff = 1.5
        decay_rate = 0.99
        
        for _ in range(self.budget // self.swarm_size):
            for i in range(self.swarm_size):
                # Evaluate the fitness of each particle
                score = func(positions[i])
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i].copy()
            
            # Update parameters adaptively
            if global_best_score < self.f_opt:
                inertia_weight *= decay_rate
            else:
                cognitive_coeff *= 1.02
                social_coeff *= 1.02
            
            # Update velocities and positions
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            for i in range(self.swarm_size):
                velocities[i] = (inertia_weight * velocities[i] +
                                 cognitive_coeff * r1 * (personal_best_positions[i] - positions[i]) +
                                 social_coeff * r2 * (global_best_position - positions[i]))
                # Clamp velocities to avoid excessive movement
                velocities[i] = np.clip(velocities[i], lb - ub, ub - lb)
                positions[i] += velocities[i]
                # Ensure particles remain within bounds
                positions[i] = np.clip(positions[i], lb, ub)
                
                # Reset velocity if stuck
                if np.allclose(velocities[i], 0, atol=1e-6):
                    velocities[i] = np.random.uniform(-1, 1, self.dim)
            
            # Update global optimum
            if global_best_score < self.f_opt:
                self.f_opt = global_best_score
                self.x_opt = global_best_position
        
        return self.f_opt, self.x_opt
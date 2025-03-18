import numpy as np
from scipy.optimize import minimize

class AdaptivePSOLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.swarm_size = min(10, budget // 3)
        self.inertia_weight = 0.5
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        
        # Initialize swarm positions and velocities
        swarm_positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        swarm_velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(swarm_positions)
        personal_best_values = np.array([func(pos) for pos in swarm_positions])
        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_index]
        global_best_value = personal_best_values[global_best_index]
        self.evaluations += self.swarm_size
        
        while self.evaluations < self.budget:
            # Update velocities and positions
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(2)
                swarm_velocities[i] = (self.inertia_weight * swarm_velocities[i] +
                                       self.cognitive_coeff * r1 * (personal_best_positions[i] - swarm_positions[i]) +
                                       self.social_coeff * r2 * (global_best_position - swarm_positions[i]))
                swarm_positions[i] = np.clip(swarm_positions[i] + swarm_velocities[i], lb, ub)
            
            # Evaluate swarm
            for i in range(self.swarm_size):
                value = func(swarm_positions[i])
                self.evaluations += 1
                if value < personal_best_values[i]:
                    personal_best_positions[i] = swarm_positions[i]
                    personal_best_values[i] = value
                    if value < global_best_value:
                        global_best_position = swarm_positions[i]
                        global_best_value = value
                
                if self.evaluations >= self.budget:
                    break
            
            # Local search exploitation
            if self.evaluations < self.budget // 2:
                res = minimize(func, global_best_position, method='L-BFGS-B',
                               bounds=[(lb[j], ub[j]) for j in range(self.dim)],
                               options={'maxiter': self.budget // 15, 'gtol': 1e-6})
                if res.fun < global_best_value:
                    global_best_position = res.x
                    global_best_value = res.fun
                
            if self.evaluations >= self.budget:
                break
        
        return global_best_position
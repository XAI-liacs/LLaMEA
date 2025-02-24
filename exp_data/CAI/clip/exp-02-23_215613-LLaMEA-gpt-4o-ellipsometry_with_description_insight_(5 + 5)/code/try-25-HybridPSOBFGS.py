import numpy as np
from scipy.optimize import minimize

class HybridPSOBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize PSO parameters
        num_particles = 10
        max_iter = self.budget // num_particles
        inertia = 0.9  # Starting inertia
        inertia_min = 0.4  # Minimum inertia value
        cognitive_coeff = 1.5
        social_coeff = 1.5
        
        # Particle positions and velocities
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        
        # Initialize personal and global best
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(p) for p in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        
        # PSO loop
        for iter_count in range(max_iter):
            for i in range(num_particles):
                # Dynamic inertia reduction
                inertia = inertia_min + (0.9 - inertia_min) * (1 - iter_count / max_iter)
                
                # Update velocity and position
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (inertia * velocities[i] +
                                 cognitive_coeff * r1 * (personal_best_positions[i] - positions[i]) +
                                 social_coeff * r2 * (global_best_position - positions[i]))
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)
                
                # Evaluate new position
                score = func(positions[i])
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]
                    
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]
            
        # Post-PSO refinement with BFGS
        result = minimize(func, global_best_position, method='L-BFGS-B', bounds=list(zip(lb, ub)))
        return result.x
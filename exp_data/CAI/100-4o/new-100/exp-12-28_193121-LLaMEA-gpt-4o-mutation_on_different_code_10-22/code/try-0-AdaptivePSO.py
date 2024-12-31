import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=50):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Initialize the swarm
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.inf)
        
        # Global best initialization
        global_best_position = None
        global_best_score = np.inf

        iter_count = 0
        while iter_count < self.budget:
            for i in range(self.swarm_size):
                # Evaluate the particle's current position
                score = func(positions[i])
                iter_count += 1
                
                # Update personal best for the particle
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]
                
                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]
                    self.f_opt, self.x_opt = global_best_score, global_best_position
                
                # Exit if budget is exhausted
                if iter_count >= self.budget:
                    break
            
            # Dynamically adjust PSO parameters based on convergence
            inertia_weight = 0.5 + np.random.rand() / 2
            cognitive_coeff = 1.5 + (global_best_score / np.max(personal_best_scores))
            social_coeff = 1.5 + (np.mean(personal_best_scores) / global_best_score)
            
            # Update velocities and positions
            for i in range(self.swarm_size):
                inertia = inertia_weight * velocities[i]
                cognitive = cognitive_coeff * np.random.rand(self.dim) * (personal_best_positions[i] - positions[i])
                social = social_coeff * np.random.rand(self.dim) * (global_best_position - positions[i])
                
                # Update velocity
                velocities[i] = inertia + cognitive + social
                
                # Update position with boundary checks
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)
        
        return self.f_opt, self.x_opt
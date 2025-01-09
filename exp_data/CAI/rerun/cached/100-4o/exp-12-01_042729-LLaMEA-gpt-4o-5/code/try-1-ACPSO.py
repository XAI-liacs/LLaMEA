import numpy as np

class ACPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 40
        self.inertia = 0.7
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        # Initialize particles
        particles = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.array([func(p) for p in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        
        evaluations = self.swarm_size
        
        while evaluations < self.budget:
            # Update velocities and positions
            r1, r2 = np.random.rand(2, self.swarm_size, self.dim)
            velocities = (
                self.inertia * velocities
                + self.cognitive_weight * r1 * (personal_best_positions - particles)
                + self.social_weight * r2 * (global_best_position - particles)
            )
            particles = np.clip(particles + velocities, self.lb, self.ub)
            
            # Evaluate new positions
            new_values = np.array([func(p) for p in particles])
            evaluations += self.swarm_size
            
            # Update personal and global bests
            better_mask = new_values < personal_best_values
            personal_best_positions[better_mask] = particles[better_mask]
            personal_best_values[better_mask] = new_values[better_mask]
            
            if np.min(personal_best_values) < global_best_value:
                global_best_position = personal_best_positions[np.argmin(personal_best_values)]
                global_best_value = np.min(personal_best_values)
                
            # Clustering-based adaptation
            if evaluations < self.budget:
                centroid = np.mean(particles, axis=0)
                perturbation = np.random.uniform(-0.1, 0.1, self.dim)
                best_cluster_center = centroid + perturbation
                
                # Use best cluster's center to guide swarm's social component
                global_best_position = np.copy(best_cluster_center)
        
        return global_best_position, global_best_value
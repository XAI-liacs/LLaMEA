import numpy as np

class DynamicPSO:
    def __init__(self, budget=10000, dim=10, n_particles=30):
        self.budget = budget
        self.dim = dim
        self.n_particles = n_particles
        self.lower_bound = -100.0
        self.upper_bound = 100.0

    def __call__(self, func):
        # Initialize particle positions and velocities
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.n_particles, self.dim))
        velocities = np.zeros((self.n_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.n_particles, np.inf)
        
        # Initialize global best
        global_best_position = None
        global_best_score = np.inf
        
        # Begin optimization
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.n_particles):
                if evaluations >= self.budget:
                    break
                
                # Evaluate the fitness of the current particle
                fitness = func(positions[i])
                evaluations += 1
                
                # Update personal best
                if fitness < personal_best_scores[i]:
                    personal_best_scores[i] = fitness
                    personal_best_positions[i] = positions[i]
                
                # Update global best
                if fitness < global_best_score:
                    global_best_score = fitness
                    global_best_position = positions[i]
            
            # Update inertia weight dynamically
            inertia_weight = 0.9 - 0.5 * (evaluations / self.budget)
            
            # Update velocities and positions
            for i in range(self.n_particles):
                r1, r2, r3 = np.random.rand(3)  # Changed line
                learning_factor_cognitive = 2.0 - 1.5 * (evaluations / self.budget)
                learning_factor_social = 2.0 + 1.5 * (evaluations / self.budget)
                
                # Simple neighborhood influence
                neighbor_indices = [(i-1) % self.n_particles, (i+1) % self.n_particles]  # Changed line
                neighbor_best_position = min(neighbor_indices, key=lambda idx: personal_best_scores[idx])  # Changed line
                
                velocities[i] = (inertia_weight * velocities[i] +
                                 learning_factor_cognitive * r1 * (personal_best_positions[i] - positions[i]) +
                                 learning_factor_social * r2 * (global_best_position - positions[i]) + 
                                 1.5 * r3 * (personal_best_positions[neighbor_best_position] - positions[i]))  # Changed line
                
                # Adaptive velocity scaling to improve convergence
                velocities[i] = np.clip(velocities[i], -20 * (1 - evaluations/self.budget), 20 * (1 - evaluations/self.budget))
                positions[i] += velocities[i]
                
                # Clamp positions to bounds
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)
        
        return global_best_score, global_best_position
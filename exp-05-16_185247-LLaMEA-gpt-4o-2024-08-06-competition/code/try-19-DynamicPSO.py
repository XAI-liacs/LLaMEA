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
                r1, r2 = np.random.rand(2)
                velocities[i] = (inertia_weight * velocities[i] +
                                 2.0 * r1 * (personal_best_positions[i] - positions[i]) +
                                 2.0 * r2 * (global_best_position - positions[i]))
                # Adaptive velocity scaling to improve convergence
                velocities[i] = np.clip(velocities[i], -20 * (1 - evaluations/self.budget), 20 * (1 - evaluations/self.budget))  
                positions[i] += velocities[i]
                
                # Clamp positions to bounds
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)
        
        return global_best_score, global_best_position
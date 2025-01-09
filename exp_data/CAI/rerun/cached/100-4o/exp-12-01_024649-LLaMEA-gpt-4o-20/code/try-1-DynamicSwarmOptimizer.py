import numpy as np

class DynamicSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30  # Number of particles in the swarm
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.w = 0.7   # Inertia weight
        self.bounds = (-5.0, 5.0)
    
    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        # Initialize particle positions and velocities
        positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.num_particles, np.inf)
        
        # Evaluate initial positions
        for i in range(self.num_particles):
            score = func(positions[i])
            personal_best_scores[i] = score

        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = np.copy(personal_best_positions[global_best_idx])

        evaluations = self.num_particles
        
        while evaluations < self.budget:
            for i in range(self.num_particles):
                # Update velocity
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (personal_best_positions[i] - positions[i])
                social = self.c2 * r2 * (global_best_position - positions[i])
                velocities[i] = self.w * velocities[i] + cognitive + social
                
                # Update position
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.bounds[0], self.bounds[1])

                # Evaluate new position
                score = func(positions[i])
                evaluations += 1

                # Check for personal best update
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = np.copy(positions[i])
                
                # Check for global best update
                if score < personal_best_scores[global_best_idx]:
                    global_best_idx = i
                    global_best_position = np.copy(personal_best_positions[global_best_idx])

                if evaluations >= self.budget:
                    break

        return global_best_position
import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30, w=0.5, c1=2.0, c2=2.0):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive (personal) coefficient
        self.c2 = c2  # Social (global) coefficient
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize particles' positions and velocities
        positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.num_particles, np.Inf)
        
        # Initialize global best variables
        global_best_position = np.zeros(self.dim)
        global_best_score = np.Inf
        
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.num_particles):
                if evaluations >= self.budget:
                    break
                score = func(positions[i])
                evaluations += 1
                
                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]
                
                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]
            
            # Update velocities and positions
            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))
            positions = positions + velocities
            
            # Enforce bounds
            positions = np.clip(positions, func.bounds.lb, func.bounds.ub)
        
        self.f_opt = global_best_score
        self.x_opt = global_best_position
        
        return self.f_opt, self.x_opt
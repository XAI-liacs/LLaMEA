import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30, c1=2.05, c2=2.05, w=0.7):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize particle positions and velocities
        positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.num_particles, np.Inf)
        
        global_best_position = None
        
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.num_particles):
                f = func(positions[i])
                evaluations += 1
                
                # Update personal best
                if f < personal_best_scores[i]:
                    personal_best_scores[i] = f
                    personal_best_positions[i] = positions[i].copy()
                
                # Update global best
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = positions[i].copy()
                    global_best_position = positions[i].copy()
                    
                if evaluations >= self.budget:
                    break

            # Update velocities and positions
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]))
                positions[i] += velocities[i]
                
                # Ensure the particles remain within bounds
                positions[i] = np.clip(positions[i], lb, ub)
        
        return self.f_opt, self.x_opt
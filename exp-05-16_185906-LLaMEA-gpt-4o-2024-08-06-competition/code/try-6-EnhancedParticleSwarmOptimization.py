import numpy as np

class EnhancedParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30, w=0.9, c1=2.0, c2=2.0):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.w = w  # initial inertia weight
        self.c1 = c1  # cognitive (personal) weight
        self.c2 = c2  # social (global) weight

    def __call__(self, func):
        # Initialize particles
        positions = np.random.uniform(-100, 100, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.num_particles, np.inf)
        
        # Evaluate initial positions
        scores = np.array([func(p) for p in positions])
        for i in range(self.num_particles):
            if scores[i] < personal_best_scores[i]:
                personal_best_scores[i] = scores[i]
                personal_best_positions[i] = positions[i]
        
        # Initialize global best
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        
        eval_count = self.num_particles
        restart_threshold = 0.1 # new threshold for stagnation restart
        
        # Main loop
        while eval_count < self.budget:
            for i in range(self.num_particles):
                # Update velocities
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.c1 * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = self.c2 * r2 * (global_best_position - positions[i])
                velocities[i] = self.w * velocities[i] + cognitive_velocity + social_velocity
                
                # Update positions
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], -100, 100)  # Ensure within bounds

            # Evaluate new positions
            scores = np.array([func(p) for p in positions])
            eval_count += self.num_particles
            
            # Update personal bests
            for i in range(self.num_particles):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = positions[i]
            
            # Update global best
            best_particle_idx = np.argmin(personal_best_scores)
            if personal_best_scores[best_particle_idx] < global_best_score:
                global_best_score = personal_best_scores[best_particle_idx]
                global_best_position = personal_best_positions[best_particle_idx]
                self.w = 0.9 # adapt inertia weight on improvement
            else:
                self.w = 0.4 # new inertia weight for stagnation

            # Random restart if stagnation occurs
            if eval_count % (self.budget // 10) == 0: # new restart strategy
                positions = np.random.uniform(-100, 100, (self.num_particles, self.dim))
        
        return global_best_score, global_best_position
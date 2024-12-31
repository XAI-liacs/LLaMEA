import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize the swarm
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.num_particles, np.Inf)
        
        # Initialize global best
        global_best_position = None
        global_best_score = np.Inf
        
        # Dynamic inertia weight parameters
        inertia_max = 0.9
        inertia_min = 0.4
        c1, c2 = 2.05, 2.05
        
        evaluations = 0

        while evaluations < self.budget:
            # Evaluate particles
            for i in range(self.num_particles):
                f = func(particles[i])
                evaluations += 1
                
                if f < personal_best_scores[i]:
                    personal_best_scores[i] = f
                    personal_best_positions[i] = particles[i]
                
                if f < global_best_score:
                    global_best_score = f
                    global_best_position = particles[i]
                
                if evaluations >= self.budget:
                    break
            
            # Calculate inertia weight
            inertia_weight = inertia_max - (inertia_max - inertia_min) * (evaluations / self.budget)
            
            # Update velocities and positions
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = c1 * r1 * (personal_best_positions[i] - particles[i])
                social_component = c2 * r2 * (global_best_position - particles[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component
                
                # Local search enhancement
                if np.random.rand() < 0.1:  # Small probability to perform local search
                    velocities[i] += np.random.normal(0, 0.1, self.dim)
                
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)
        
        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt
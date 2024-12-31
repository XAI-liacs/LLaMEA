import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, swarm_size=50):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize particles
        particles = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(self.swarm_size, np.Inf)
        
        for eval_count in range(0, self.budget, self.swarm_size):
            # Evaluate function for each particle
            for i in range(self.swarm_size):
                f = func(particles[i])
                if f < personal_best_scores[i]:
                    personal_best_scores[i] = f
                    personal_best_positions[i] = particles[i].copy()
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = particles[i].copy()
            
            # Find global best position
            global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
            
            # Update velocities and positions
            inertia = 0.5 + (0.9 - 0.5) * (1 - eval_count / self.budget)
            for i in range(self.swarm_size):
                cognitive_component = np.random.rand(self.dim) * (personal_best_positions[i] - particles[i])
                social_component = np.random.rand(self.dim) * (global_best_position - particles[i])
                velocities[i] = inertia * velocities[i] + 2.0 * cognitive_component + 2.0 * social_component
                particles[i] = np.clip(particles[i] + velocities[i], func.bounds.lb, func.bounds.ub)
        
        return self.f_opt, self.x_opt
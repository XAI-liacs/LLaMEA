import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30, inertia=0.5, cognitive_weight=1.5, social_weight=2.0):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia = inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        personal_best_positions = particles.copy()
        personal_best_values = np.full(self.num_particles, np.Inf)
        global_best_value = np.Inf
        global_best_position = None
        
        for i in range(self.budget):
            for j in range(self.num_particles):
                f = func(particles[j])
                
                if f < personal_best_values[j]:
                    personal_best_values[j] = f
                    personal_best_positions[j] = particles[j]
                
                if f < global_best_value:
                    global_best_value = f
                    global_best_position = particles[j]
            
            for j in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[j] = self.inertia * velocities[j] + \
                                self.cognitive_weight * r1 * (personal_best_positions[j] - particles[j]) + \
                                self.social_weight * r2 * (global_best_position - particles[j])
                
                particles[j] = particles[j] + velocities[j]
                particles[j] = np.clip(particles[j], lb, ub)
        
        self.f_opt = global_best_value
        self.x_opt = global_best_position
        
        return self.f_opt, self.x_opt
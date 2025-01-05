import numpy as np

class ImprovedParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30, inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        particles_pos = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        particles_vel = np.zeros((self.num_particles, self.dim))
        personal_best_pos = np.copy(particles_pos)
        personal_best_val = np.array([func(x) for x in personal_best_pos])
        global_best_idx = np.argmin(personal_best_val)
        global_best_val = personal_best_val[global_best_idx]
        global_best_pos = np.copy(personal_best_pos[global_best_idx])

        for _ in range(self.budget):
            inertia_weight = 0.4 + 0.5*(1 - _/self.budget)  # Update inertia weight dynamically
            for i in range(self.num_particles):
                new_vel = inertia_weight * particles_vel[i] + \
                          self.cognitive_weight * np.random.rand() * (personal_best_pos[i] - particles_pos[i]) + \
                          self.social_weight * np.random.rand() * (global_best_pos - particles_pos[i])
                particles_vel[i] = new_vel
                particles_pos[i] = np.clip(particles_pos[i] + new_vel, lb, ub)
                
                f = func(particles_pos[i])
                if f < personal_best_val[i]:
                    personal_best_val[i] = f
                    personal_best_pos[i] = particles_pos[i]
                    if f < global_best_val:
                        global_best_val = f
                        global_best_pos = particles_pos[i]

        self.f_opt = global_best_val
        self.x_opt = global_best_pos

        return self.f_opt, self.x_opt
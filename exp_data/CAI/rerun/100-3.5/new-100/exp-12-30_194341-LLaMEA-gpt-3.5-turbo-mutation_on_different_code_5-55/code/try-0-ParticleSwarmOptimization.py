import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30, inertia_weight=0.7, cognitive_weight=1.5, social_weight=1.5):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        particles_pos = np.random.uniform(bounds[0], bounds[1], size=(self.num_particles, self.dim))
        particles_vel = np.zeros((self.num_particles, self.dim))
        particles_best_pos = particles_pos.copy()
        particles_best_val = np.full(self.num_particles, np.Inf)
        global_best_val = np.Inf
        global_best_pos = None

        for _ in range(self.budget):
            for i in range(self.num_particles):
                fitness = func(particles_pos[i])
                if fitness < particles_best_val[i]:
                    particles_best_val[i] = fitness
                    particles_best_pos[i] = particles_pos[i]
                
                if fitness < global_best_val:
                    global_best_val = fitness
                    global_best_pos = particles_pos[i]
                
                r1, r2 = np.random.random(size=(2, self.dim))
                particles_vel[i] = self.inertia_weight * particles_vel[i] + \
                                   self.cognitive_weight * r1 * (particles_best_pos[i] - particles_pos[i]) + \
                                   self.social_weight * r2 * (global_best_pos - particles_pos[i])
                particles_pos[i] = np.clip(particles_pos[i] + particles_vel[i], bounds[0], bounds[1])

        self.f_opt = global_best_val
        self.x_opt = global_best_pos

        return self.f_opt, self.x_opt
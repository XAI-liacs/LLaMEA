import numpy as np

class EnhancedParticleAccelerationOptimization:
    def __init__(self, budget, dim, num_particles=30, inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5, acceleration_coefficient=2.0, inertia_decay=0.99):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.acceleration_coefficient = acceleration_coefficient
        self.inertia_decay = inertia_decay

    def __call__(self, func):
        def initialize_particles():
            return np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))

        particles = initialize_particles()
        velocities = np.zeros((self.num_particles, self.dim))
        personal_bests = particles.copy()
        global_best = particles[np.argmin([func(p) for p in particles])]

        for _ in range(self.budget):
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                adaptive_inertia = self.inertia_weight * self.inertia_decay
                velocities[i] = adaptive_inertia * velocities[i] + \
                                self.cognitive_weight * r1 * (personal_bests[i] - particles[i]) + \
                                self.social_weight * r2 * (global_best - particles[i])
                particles[i] = particles[i] + velocities[i]

                if func(particles[i]) < func(personal_bests[i]):
                    personal_bests[i] = particles[i]
                    if func(particles[i]) < func(global_best):
                        global_best = particles[i]

        return global_best
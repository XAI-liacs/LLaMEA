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
        particles = np.random.uniform(lb, ub, size=(self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        pbest_positions = particles.copy()
        pbest_values = np.full(self.num_particles, np.inf)
        gbest_position = None
        gbest_value = np.inf

        for _ in range(self.budget):
            for i in range(self.num_particles):
                f = func(particles[i])
                if f < pbest_values[i]:
                    pbest_values[i] = f
                    pbest_positions[i] = particles[i]
                if f < gbest_value:
                    gbest_value = f
                    gbest_position = particles[i]

                inertia_term = self.inertia * velocities[i]
                cognitive_term = self.cognitive_weight * np.random.rand() * (pbest_positions[i] - particles[i])
                social_term = self.social_weight * np.random.rand() * (gbest_position - particles[i])

                velocities[i] = inertia_term + cognitive_term + social_term
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

        self.f_opt = gbest_value
        self.x_opt = gbest_position
        return self.f_opt, self.x_opt
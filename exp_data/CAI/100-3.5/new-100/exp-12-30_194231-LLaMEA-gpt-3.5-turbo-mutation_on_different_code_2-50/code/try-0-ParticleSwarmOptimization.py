import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30, inertia=0.5, cognitive_rate=1.5, social_rate=2.0):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia = inertia
        self.cognitive_rate = cognitive_rate
        self.social_rate = social_rate
        self.bounds = (-5.0, 5.0)
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        particles = np.random.uniform(self.bounds[0], self.bounds[1], size=(self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))

        pbest_positions = particles.copy()
        pbest_values = np.array([func(p) for p in particles])

        gbest_index = np.argmin(pbest_values)
        gbest_value = pbest_values[gbest_index]
        gbest_position = pbest_positions[gbest_index].copy()

        for _ in range(self.budget):
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = self.inertia * velocities[i] + self.cognitive_rate * r1 * (pbest_positions[i] - particles[i]) + self.social_rate * r2 * (gbest_position - particles[i])
                particles[i] = np.clip(particles[i] + velocities[i], self.bounds[0], self.bounds[1])

                current_value = func(particles[i])
                if current_value < pbest_values[i]:
                    pbest_values[i] = current_value
                    pbest_positions[i] = particles[i].copy()

                    if current_value < gbest_value:
                        gbest_value = current_value
                        gbest_position = particles[i].copy()

        self.f_opt = gbest_value
        self.x_opt = gbest_position

        return self.f_opt, self.x_opt
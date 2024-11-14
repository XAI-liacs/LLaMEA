import numpy as np

class EnhancedQPSO:
    def __init__(self, budget, dim, num_particles=30, inertia_weight=0.9, cognitive_weight=2.0, social_weight=2.0):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

    def __call__(self, func):
        def initialize_particles():
            return np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))

        def update_position(particles, velocities):
            return np.clip(particles + velocities, -5.0, 5.0)

        def evaluate_fitness(particles):
            return np.array([func(p) for p in particles])

        particles = initialize_particles()
        best_global_position = particles[np.argmin(evaluate_fitness(particles))]
        velocities = np.zeros_like(particles)

        for _ in range(self.budget):
            for i in range(self.num_particles):
                rand1 = np.random.rand(self.dim)
                rand2 = np.random.rand(self.dim)
                adaptive_inertia = self.inertia_weight - (self.inertia_weight / self.budget) * _
                velocities[i] = adaptive_inertia * velocities[i] + \
                                self.cognitive_weight * rand1 * (best_global_position - particles[i]) + \
                                self.social_weight * rand2 * (best_global_position - particles[i])

                particles[i] = update_position(particles[i], velocities[i])

            fitness_values = evaluate_fitness(particles)
            best_particle_index = np.argmin(fitness_values)
            if fitness_values[best_particle_index] < func(best_global_position):
                best_global_position = particles[best_particle_index]
        
        if _ % 10 == 0 and _ != 0:
            self.num_particles = int(self.num_particles * 1.1)  # Increase particle number dynamically every 10 iterations

        return best_global_position
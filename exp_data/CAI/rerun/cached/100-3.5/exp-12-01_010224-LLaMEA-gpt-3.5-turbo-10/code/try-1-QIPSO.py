import numpy as np

class QIPSO:
    def __init__(self, budget, dim, num_particles=30, omega=0.5, phi_p=0.5, phi_g=0.5):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.omega = omega
        self.phi_p = phi_p
        self.phi_g = phi_g

    def __call__(self, func):
        def quantum_rotation(particle, global_best):
            theta = np.arccos(np.dot(particle, global_best) / (np.linalg.norm(particle) * np.linalg.norm(global_best)))
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            return np.dot(rotation_matrix, particle)

        def objective_function(x):
            return func(x)

        def initialize_particles():
            return np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))

        particles = initialize_particles()
        best_particle = particles[np.argmin([objective_function(p) for p in particles])]
        global_best = best_particle.copy()

        for _ in range(self.budget):
            for i in range(self.num_particles):
                r_p, r_g = np.random.uniform(0, 1, size=2)
                velocity = self.omega * particles[i] + self.phi_p * r_p * (best_particle - particles[i]) + self.phi_g * r_g * (global_best - particles[i])
                particles[i] = quantum_rotation(velocity, global_best)
                if objective_function(particles[i]) < objective_function(best_particle):
                    best_particle = particles[i].copy()
                if objective_function(particles[i]) < objective_function(global_best):
                    global_best = particles[i].copy()

        return global_best
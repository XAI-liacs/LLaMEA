import numpy as np

class QE_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(50, budget // 2)
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.w = 0.7   # Inertia weight
        self.vel_limit = 0.1
        self.particles = None
        self.velocities = None
        self.pbest_positions = None
        self.pbest_values = None
        self.gbest_position = None
        self.gbest_value = np.inf

    def initialize_particles(self, lb, ub):
        self.particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-self.vel_limit, self.vel_limit, (self.num_particles, self.dim))
        self.pbest_positions = self.particles.copy()
        self.pbest_values = np.full(self.num_particles, np.inf)

    def quantum_update(self, lb, ub):
        phi = np.random.uniform(0, 2 * np.pi, (self.num_particles, self.dim))
        radius = np.random.uniform(0, 1, self.num_particles)
        quantum_positions = (self.gbest_position + radius[:, None] * (
            self.pbest_positions - self.gbest_position) * np.cos(phi))
        return np.clip(quantum_positions, lb, ub)

    def update_velocities_and_positions(self, lb, ub):
        r1 = np.random.rand(self.num_particles, self.dim)
        r2 = np.random.rand(self.num_particles, self.dim)
        cognitive = self.c1 * r1 * (self.pbest_positions - self.particles)
        social = self.c2 * r2 * (self.gbest_position - self.particles)
        self.velocities = self.w * self.velocities + cognitive + social
        self.velocities = np.clip(self.velocities, -self.vel_limit, self.vel_limit)
        self.particles += self.velocities
        self.particles = np.clip(self.particles, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_particles(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            quantum_positions = self.quantum_update(lb, ub)
            self.update_velocities_and_positions(lb, ub)

            for i in range(self.num_particles):
                if evaluations >= self.budget:
                    break

                # Evaluate quantum positions
                quantum_value = func(quantum_positions[i])
                if quantum_value < self.pbest_values[i]:
                    self.pbest_values[i] = quantum_value
                    self.pbest_positions[i] = quantum_positions[i].copy()

                # Evaluate particle positions
                particle_value = func(self.particles[i])
                evaluations += 2  # counting both quantum and particle evaluations

                if particle_value < self.pbest_values[i]:
                    self.pbest_values[i] = particle_value
                    self.pbest_positions[i] = self.particles[i].copy()

                if particle_value < self.gbest_value:
                    self.gbest_value = particle_value
                    self.gbest_position = self.particles[i].copy()

        return self.gbest_position, self.gbest_value
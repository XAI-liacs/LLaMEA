import numpy as np

class QIGPO:
    def __init__(self, budget, dim, num_particles=30, alpha=0.5, mutation_rate=0.1, crossover_rate=0.8):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.alpha = alpha
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')

        # Initialize particle positions using quantum-inspired approach
        particles = self.initialize_particles(lb, ub)
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))

        while self.evaluations < self.budget:
            for i in range(self.num_particles):
                # Update particle velocity and position
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = self.alpha * velocities[i] + r1 * (best_global_position - particles[i] if best_global_position is not None else 0) + r2 * (ub - lb) * self.quantum_superposition()
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

                # Evaluate particle
                value = func(particles[i])
                self.evaluations += 1

                # Update global best
                if value < best_global_value:
                    best_global_value = value
                    best_global_position = particles[i]

                if self.evaluations >= self.budget:
                    break

            # Perform genetic operations
            self.genetic_operations(particles, lb, ub)

        return best_global_position

    def initialize_particles(self, lb, ub):
        # Quantum-inspired superposition for initialization
        return lb + (ub - lb) * np.random.rand(self.num_particles, self.dim)

    def quantum_superposition(self):
        # Simulate quantum superposition for additional exploration
        return np.random.choice([-1, 1], self.dim) * np.random.rand(self.dim)

    def genetic_operations(self, particles, lb, ub):
        # Mutation and crossover for enhanced exploration
        for i in range(self.num_particles):
            if np.random.rand() < self.mutation_rate:
                mutation = (np.random.rand(self.dim) - 0.5) * 0.1 * (ub - lb)
                particles[i] = np.clip(particles[i] + mutation, lb, ub)
        
        for i in range(0, self.num_particles - 1, 2):
            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, self.dim)
                particles[i][:crossover_point], particles[i+1][:crossover_point] = (
                    particles[i+1][:crossover_point].copy(), particles[i][:crossover_point].copy())
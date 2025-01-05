import numpy as np

class QuantumParticle:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds.lb, bounds.ub, dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')

class QuantumPSO:
    def __init__(self, budget, dim, swarm_size=30, alpha=0.5, beta=0.9):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.alpha = alpha
        self.beta = beta
        self.particles = []
        self.global_best_position = None
        self.global_best_value = float('inf')

    def __call__(self, func):
        # Initialize particles
        bounds = func.bounds
        self.particles = [QuantumParticle(self.dim, bounds) for _ in range(self.swarm_size)]
        eval_count = 0
        inertia_weight = 0.9  # Line modified for adaptive inertia weight

        while eval_count < self.budget:
            for particle in self.particles:
                # Evaluate the fitness function
                fitness_value = func(particle.position)
                eval_count += 1

                # Update personal best
                if fitness_value < particle.best_value:
                    particle.best_value = fitness_value
                    particle.best_position = np.copy(particle.position)

                # Update global best
                if fitness_value < self.global_best_value:
                    self.global_best_value = fitness_value
                    self.global_best_position = np.copy(particle.position)

            # Update particle positions and velocities
            for particle in self.particles:
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.alpha * r1 * (particle.best_position - particle.position)
                social_component = self.beta * r2 * (self.global_best_position - particle.position)

                # Quantum-inspired velocity update with adaptive inertia
                particle.velocity = inertia_weight * particle.velocity + cognitive_component + social_component

                # Quantum tunneling: Adaptive jump to avoid local maxima
                if np.random.rand() < 0.1:
                    tunneling_strength = np.random.uniform(0.1, 1.0)
                    particle.position += tunneling_strength * np.random.uniform(bounds.lb, bounds.ub, self.dim)
                else:
                    particle.position += particle.velocity

                # Ensure position stays within bounds
                particle.position = np.clip(particle.position, bounds.lb, bounds.ub)

            if eval_count >= self.budget:
                break

        return self.global_best_position
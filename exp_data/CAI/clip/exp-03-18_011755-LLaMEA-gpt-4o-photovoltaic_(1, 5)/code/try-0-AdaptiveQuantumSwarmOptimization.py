import numpy as np

class AdaptiveQuantumSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 2)
        self.particles = np.random.uniform(size=(self.population_size, dim))
        self.velocities = np.random.uniform(size=(self.population_size, dim))
        self.personal_best = np.copy(self.particles)
        self.global_best = None
        self.best_value = float('inf')
        self.evaluations = 0

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                position = self.particles[i]
                # Quantum-inspired position update
                quantum_factor = np.random.uniform(size=self.dim)
                new_position = (
                    position
                    + np.sin(quantum_factor * np.pi) * self.velocities[i]
                    + np.cos(quantum_factor * np.pi) * (self.global_best - position)
                )
                new_position = np.clip(new_position, bounds[0], bounds[1])

                # Evaluate new position
                value = func(new_position)
                self.evaluations += 1

                # Update personal and global bests
                if value < self.best_value:
                    self.global_best = new_position
                    self.best_value = value
                if value < func(self.personal_best[i]):
                    self.personal_best[i] = new_position

                # Velocity update
                inertia = 0.5 + np.random.rand() / 2
                cognitive = np.random.rand() * (self.personal_best[i] - position)
                social = np.random.rand() * (self.global_best - position)
                self.velocities[i] = inertia * self.velocities[i] + cognitive + social

                # Update particle position
                self.particles[i] = new_position

        return self.global_best
import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particles = np.random.uniform(-5.0, 5.0, (10, dim))
        self.best_position = self.particles[np.argmin([func(x) for x in self.particles])]
        self.best_value = func(self.best_position)
        self.temperature = 1.0
        self.alpha = 0.95

    def __call__(self, func):
        for _ in range(self.budget):
            for particle in self.particles:
                new_particle = particle + np.random.normal(0, 1, self.dim)
                new_particle = np.clip(new_particle, -5.0, 5.0)
                new_value = func(new_particle)
                if new_value < func(particle):
                    particle[:] = new_particle
                    if new_value < self.best_value:
                        self.best_position = new_particle
                        self.best_value = new_value
                else:
                    if np.random.rand() < np.exp((func(particle) - new_value) / self.temperature):
                        particle[:] = new_particle
            self.temperature *= self.alpha
        return self.best_position
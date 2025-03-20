import numpy as np

class Particle:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds.lb, bounds.ub, dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')

class ParticleSwarmDifferential:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particles = []
        self.bounds = None
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive (particle) weight
        self.c2 = 1.5  # social (swarm) weight

    def __call__(self, func):
        self.bounds = func.bounds
        num_particles = min(40, self.budget // 5)  # adaptively choosing swarm size
        self.particles = [Particle(self.dim, self.bounds) for _ in range(num_particles)]
        
        for t in range(self.budget):
            self.w = 0.9 - t * ((0.9 - 0.4) / self.budget)  # dynamically adjust inertia weight
            for particle in self.particles:
                fitness = func(particle.position)
                if fitness < particle.best_value:
                    particle.best_value = fitness
                    particle.best_position = np.copy(particle.position)

                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = np.copy(particle.position)

                # Update velocity
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                particle.velocity = (self.w * particle.velocity +
                                     self.c1 * r1 * (particle.best_position - particle.position) +
                                     self.c2 * r2 * (self.global_best_position - particle.position))
                # Perform differential mutation
                a, b, c = np.random.choice(self.particles, 3, replace=False)
                F = np.random.uniform(0.5, 0.9)
                mutant = a.position + F * (b.position - c.position)
                mutant = np.clip(mutant, self.bounds.lb, self.bounds.ub)

                trial = np.copy(particle.position)
                for j in range(self.dim):
                    if np.random.rand() < 0.5:
                        trial[j] = mutant[j]

                trial_fitness = func(trial)
                if trial_fitness < fitness:
                    particle.position = trial
                else:
                    particle.position += particle.velocity

        return self.global_best_position, self.global_best_value
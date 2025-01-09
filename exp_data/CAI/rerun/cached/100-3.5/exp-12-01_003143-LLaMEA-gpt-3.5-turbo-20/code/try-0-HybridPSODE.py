import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim, swarm_size=30, f=0.5, cr=0.9, w=0.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f = f
        self.cr = cr
        self.w = w

    def __call__(self, func):
        def create_particle():
            return np.random.uniform(-5.0, 5.0, self.dim), np.inf

        def mutate(particles, best, f, cr):
            p_best, _ = best
            new_particles = []
            for particle, _ in particles:
                idxs = np.random.choice(len(particles), 3, replace=False)
                x1, x2, x3 = particles[idxs]
                mutant = np.clip(x1 + f * (x2 - x3), -5.0, 5.0)
                crossover = np.random.rand(self.dim) < cr
                trial = np.where(crossover, mutant, particle)
                new_particles.append((trial, func(trial)))
            return new_particles

        particles = [create_particle() for _ in range(self.swarm_size)]
        global_best = min(particles, key=lambda x: x[1])

        for _ in range(self.budget):
            new_particles = mutate(particles, global_best, self.f, self.cr)
            particles = sorted(new_particles, key=lambda x: x[1])[:self.swarm_size]
            global_best = min(global_best, min(particles, key=lambda x: x[1]), key=lambda x: x[1])

        return global_best[0]
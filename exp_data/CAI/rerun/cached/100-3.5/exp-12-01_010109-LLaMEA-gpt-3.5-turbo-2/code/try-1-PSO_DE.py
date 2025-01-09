import numpy as np

class PSO_DE:
    def __init__(self, budget, dim, swarm_size=30, pso_w=0.5, pso_c1=1.5, pso_c2=1.5, de_f=0.8, de_cr=0.9):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.pso_w = pso_w
        self.pso_c1 = pso_c1
        self.pso_c2 = pso_c2
        self.de_f = de_f
        self.de_cr = de_cr

    def __call__(self, func):
        def evaluate_particle(particle):
            return func(particle)

        def initialize_swarm():
            return np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))

        def optimize_swarm(swarm):
            for i in range(self.budget):
                for idx, particle in enumerate(swarm):
                    p_best = swarm[np.argmin([evaluate_particle(p) for p in swarm])]
                    g_best = p_best if evaluate_particle(p_best) < evaluate_particle(swarm[idx]) else swarm[idx]

                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    new_velocity = self.pso_w * swarm[idx] + self.pso_c1 * r1 * (p_best - swarm[idx]) + self.pso_c2 * r2 * (g_best - swarm[idx])

                    mutant = swarm[np.random.choice(np.delete(np.arange(self.swarm_size), idx, 0), 3, replace=False)]
                    trial = swarm[idx] + self.de_f * (mutant[0] - mutant[1]) + self.de_f * (mutant[2] - mutant[3])

                    crossover_mask = np.random.rand(self.dim) < self.de_cr
                    swarm[idx] = np.where(crossover_mask, trial, new_velocity)

            return swarm[np.argmin([evaluate_particle(p) for p in swarm])]

        swarm = initialize_swarm()
        return optimize_swarm(swarm)
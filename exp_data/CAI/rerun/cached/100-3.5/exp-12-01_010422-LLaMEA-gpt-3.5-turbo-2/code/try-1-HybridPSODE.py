import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def within_bounds(x):
            return np.clip(x, -5.0, 5.0)

        def evaluate(x):
            return func(within_bounds(x))

        swarm_size = 10
        pso_max_iter = int(self.budget / 2)
        de_max_iter = int(self.budget / 2)
        pso_w = 0.5
        pso_c1 = 1.5
        pso_c2 = 1.5

        # Initialize particles randomly within the search space
        particles = np.random.uniform(-5.0, 5.0, (swarm_size, self.dim))
        velocities = np.zeros((swarm_size, self.dim))
        pbest = particles.copy()

        # PSO optimization
        for _ in range(pso_max_iter):
            for i in range(swarm_size):
                if evaluate(particles[i]) < evaluate(pbest[i]):
                    pbest[i] = particles[i]
            gbest = pbest[np.argmin([evaluate(p) for p in pbest])]

            for i in range(swarm_size):
                r1, r2 = np.random.uniform(0, 1, 2)
                velocities[i] = pso_w * velocities[i] + pso_c1 * r1 * (pbest[i] - particles[i]) + pso_c2 * r2 * (gbest - particles[i])
                particles[i] = within_bounds(particles[i] + velocities[i])

        # DE optimization
        bounds = [(-5.0, 5.0)] * self.dim
        for _ in range(de_max_iter):
            for i in range(swarm_size):
                mutant = within_bounds(particles[i] + 0.8 * (particles[np.random.choice(range(swarm_size))] - particles[np.random.choice(range(swarm_size))]))
                trial = np.where(np.random.uniform(0, 1, self.dim) < 0.5, mutant, particles[i])
                if evaluate(trial) < evaluate(particles[i]):
                    particles[i] = trial

        return evaluate(gbest)
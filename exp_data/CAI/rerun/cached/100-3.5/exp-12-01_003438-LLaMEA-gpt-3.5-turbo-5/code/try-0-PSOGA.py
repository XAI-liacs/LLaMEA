import numpy as np

class PSOGA:
    def __init__(self, budget, dim, swarm_size=30, mutation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.mutation_rate = mutation_rate

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
        velocity = np.zeros((self.swarm_size, self.dim))
        pbest = swarm.copy()
        gbest = pbest[np.argmin([func(ind) for ind in pbest])]
        
        for _ in range(self.budget):
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocity[i] = 0.5 * velocity[i] + 1.5 * r1 * (pbest[i] - swarm[i]) + 2.0 * r2 * (gbest - swarm[i])
                swarm[i] += velocity[i]
                if np.random.rand() < self.mutation_rate:
                    swarm[i] += np.random.normal(0, 1, size=self.dim)
                pbest[i] = np.where(func(swarm[i]) < func(pbest[i]), swarm[i], pbest[i])
                gbest = pbest[np.argmin([func(ind) for ind in pbest])]
        
        return gbest
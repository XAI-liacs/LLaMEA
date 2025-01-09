import numpy as np

class HybridPSODWS:
    def __init__(self, budget, dim, w=0.7, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.dim, self.dim))
        velocity = np.zeros((self.dim, self.dim))

        for _ in range(self.budget):
            for i in range(self.dim):
                r1, r2 = np.random.rand(), np.random.rand()
                velocity[i] = self.w * velocity[i] + self.c1 * r1 * (swarm[i] - swarm[i]) + self.c2 * r2 * (swarm[i] - swarm[i])
                swarm[i] = swarm[i] + velocity[i]

        fitness = [func(individual) for individual in swarm]
        best_index = np.argmin(fitness)
        best_solution = swarm[best_index]
        
        return best_solution
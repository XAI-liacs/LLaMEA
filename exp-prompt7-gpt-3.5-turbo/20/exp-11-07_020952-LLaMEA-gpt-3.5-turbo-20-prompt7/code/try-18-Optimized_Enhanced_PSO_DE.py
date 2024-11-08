import numpy as np

class Optimized_Enhanced_PSO_DE:
    def __init__(self, budget, dim, swarm_size=30, pso_weight=0.5, c1=1.5, c2=2.0, de_weight=0.8, de_cr=0.7):
        self.budget, self.dim, self.swarm_size = budget, dim, swarm_size
        self.pso_weight, self.c1, self.c2, self.de_weight, self.de_cr = pso_weight, c1, c2, de_weight, de_cr
        self.rand_pso = self.pso_weight * np.random.uniform(0, 1, size=(self.swarm_size, self.dim))
        self.rand_de = self.de_weight * np.random.uniform(0, 1, size=(self.swarm_size, self.dim))

    def __call__(self, func):
        def limit_bounds(x):
            return np.clip(x, -5.0, 5.0)

        swarm = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
        velocity = np.zeros((self.swarm_size, self.dim))
        pbest = swarm.copy()
        gbest = min(pbest, key=func)

        for _ in range(self.budget):
            for i in range(self.swarm_size):
                x, v, p = swarm[i], velocity[i], pbest[i]
                r1, r2 = np.random.uniform(0, 1, size=(2, self.dim))
                r1_c1 = self.c1 * r1
                r2_c2 = self.c2 * r2

                swarm[i], velocity[i] = limit_bounds(x + self.rand_pso[i] + r1_c1 * (p - x) + r2_c2 * (gbest - x)), v

                r1, r2, r3 = np.random.choice(self.swarm_size, 3, replace=False)
                p1_p2 = pbest[r1]
                p2_p3 = pbest[r2] - pbest[r3]
                rand_de_pbest = self.rand_de[r1]

                trial = limit_bounds(p1_p2 + rand_de_pbest * p2_p3)
                pbest[i] = trial if func(trial) < func(p) else p

            gbest = min(pbest, key=func)

        return gbest
import numpy as np

class Enhanced_Efficient_PSO_DE_Optimizer_Improved:
    def __init__(self, budget, dim, swarm_size=30, de_cr=0.5, de_f=0.8, w=0.5, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.de_f = de_f
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def __call__(self, func):
        def pso_de_optimize():
            swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
            velocity = np.zeros((self.swarm_size, self.dim))
            pbest = swarm.copy()
            pbest_fitness = np.array([func(ind) for ind in swarm])
            gbest_idx = np.argmin(pbest_fitness)
            gbest = swarm[gbest_idx]

            for _ in range(self.budget):
                r1_r2 = np.random.rand(2, self.swarm_size, self.dim)
                velocity = self.w * velocity + self.c1 * r1_r2[0] * (pbest - swarm) + self.c2 * r1_r2[1] * (gbest - swarm)
                swarm = np.clip(swarm + velocity, -5.0, 5.0)

                trial_vectors = np.clip(swarm + np.random.choice(swarm.shape[0], (3, swarm.shape[0]), replace=True) * self.de_f * (swarm - swarm), -5.0, 5.0)
                trial_fitness = np.array([func(tv) for tv in trial_vectors])

                updates = trial_fitness < pbest_fitness
                pbest = np.where(updates[:, None], trial_vectors, pbest)
                pbest_fitness = np.where(updates, trial_fitness, pbest_fitness)

                gbest_idx = np.argmin(pbest_fitness)
                gbest = pbest[gbest_idx]

            return gbest

        return pso_de_optimize()
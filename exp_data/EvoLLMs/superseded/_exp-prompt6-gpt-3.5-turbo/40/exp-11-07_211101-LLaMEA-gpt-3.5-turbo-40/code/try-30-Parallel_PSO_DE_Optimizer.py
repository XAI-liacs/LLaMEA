import numpy as np
from concurrent.futures import ThreadPoolExecutor

class Parallel_PSO_DE_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, de_cr=0.5, de_f=0.8, w=0.5, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.de_cr = de_cr
        self.de_f = de_f
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def __call__(self, func):
        def de_mutate(pop, target_idx):
            candidates = np.delete(np.arange(len(pop)), target_idx)
            a, b, c = pop[np.random.choice(candidates, 3, replace=False)]
            return np.clip(a + self.de_f * (b - c), -5.0, 5.0)

        def evaluate_fitness(pop):
            return [func(ind) for ind in pop]

        def pso_de_optimize():
            swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
            velocity = np.zeros((self.swarm_size, self.dim))
            pbest = swarm.copy()
            pbest_fitness = np.array(evaluate_fitness(swarm))
            gbest_idx = np.argmin(pbest_fitness)
            gbest = swarm[gbest_idx]

            with ThreadPoolExecutor() as executor:
                for _ in range(self.budget):
                    r1_r2 = np.random.rand(2, self.swarm_size, self.dim)
                    velocity = self.w * velocity + self.c1 * r1_r2[0] * (pbest - swarm) + self.c2 * r1_r2[1] * (gbest - swarm)
                    swarm = np.clip(swarm + velocity, -5.0, 5.0)

                    trial_vectors = np.array([de_mutate(swarm, i) for i in range(self.swarm_size)])
                    trial_fitness = np.array(executor.map(func, trial_vectors))

                    updates = trial_fitness < pbest_fitness
                    pbest[updates] = trial_vectors[updates]
                    pbest_fitness[updates] = trial_fitness[updates]

                    gbest_idx = np.argmin(pbest_fitness)
                    gbest = pbest[gbest_idx]

            return gbest

        return pso_de_optimize()
import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim, swarm_size=30, de_pop_size=10, de_mut_prob=0.8, de_cross_prob=0.7):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.de_pop_size = de_pop_size
        self.de_mut_prob = de_mut_prob
        self.de_cross_prob = de_cross_prob

    def __call__(self, func):
        def fitness(individual):
            return func(individual)

        swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        personal_best = swarm.copy()
        global_best = swarm[np.argmin([fitness(x) for x in swarm])]

        for _ in range(self.budget):
            for i in range(self.swarm_size):
                p = np.random.uniform(0, 1, self.dim)
                v = np.random.uniform(-1, 1, self.dim)
                swarm[i] = swarm[i] + p * (personal_best[i] - swarm[i]) + v * (global_best - swarm[i])
                swarm[i] = np.clip(swarm[i], -5.0, 5.0)
                if fitness(swarm[i]) < fitness(personal_best[i]):
                    personal_best[i] = swarm[i]
                    if fitness(swarm[i]) < fitness(global_best):
                        global_best = swarm[i]

            de_population = np.array([np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.de_pop_size)])
            for j in range(self.de_pop_size):
                idxs = np.random.choice(self.swarm_size, 3, replace=False)
                mutant = swarm[idxs[0]] + self.de_mut_prob * (swarm[idxs[1]] - swarm[idxs[2]])
                cross_points = np.random.rand(self.dim) < self.de_cross_prob
                trial = de_population[j].copy()
                trial[cross_points] = mutant[cross_points]
                if fitness(trial) < fitness(de_population[j]):
                    de_population[j] = trial

            swarm = np.vstack((swarm, de_population))
            swarm = swarm[np.argsort([fitness(x) for x in swarm])[:self.swarm_size]]

        return global_best
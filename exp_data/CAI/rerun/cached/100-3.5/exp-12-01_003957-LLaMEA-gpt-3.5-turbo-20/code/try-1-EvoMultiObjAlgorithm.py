import numpy as np
from scipy.stats import rankdata

class EvoMultiObjAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.cr = 0.5
        self.f = 0.5
        self.w = 0.7

    def __call__(self, func):
        def mutate(x, pop, idx, f):
            a, b, c = np.random.choice(pop, 3, replace=False)
            mutant = np.clip(x + f * (a - x + b - c), -5.0, 5.0)
            return mutant

        def crossover(x, mutant, cr):
            crossover_mask = np.random.rand(self.dim) < cr
            trial = np.where(crossover_mask, mutant, x)
            return trial

        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        for _ in range(self.budget):
            rank = rankdata(fitness, method='ordinal')
            ranks = len(rank) - rank + 1
            weights = ranks / np.sum(ranks)
            gbest = pop[np.argmax(weights)]

            for i in range(self.pop_size):
                mutant = mutate(pop[i], pop, i, self.f)
                trial = crossover(pop[i], mutant, self.cr)
                v = self.w * velocities[i] + np.random.uniform() * (gbest - pop[i]) + np.random.uniform() * (trial - pop[i])
                velocities[i] = v
                pop[i] = np.clip(pop[i] + v, -5.0, 5.0)
                fitness[i] = func(pop[i])
        return pop[np.argmin(fitness)]
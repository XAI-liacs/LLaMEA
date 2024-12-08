import numpy as np
import random

class HyperElitistSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.elite_size = int(budget * 0.2)
        self.swarm_size = budget - elite_size
        self.elite = np.zeros((self.elite_size, self.dim))
        self.swarm = np.zeros((self.swarm_size, self.dim))
        self.best = np.inf
        self.refine_prob = 0.15

    def __call__(self, func):
        for _ in range(self.budget):
            # Initialize swarm and elite
            self.swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
            self.elite = np.random.uniform(-5.0, 5.0, (self.elite_size, self.dim))

            # Evaluate function for each individual
            for i in range(self.swarm_size + self.elite_size):
                x = np.zeros(self.dim)
                if i < self.swarm_size:
                    x = self.swarm[i]
                else:
                    x = self.elite[i - self.swarm_size]
                f = func(x)
                if f < self.best:
                    self.best = f
                    self.elite[i - self.swarm_size] = x
                if f < func(self.elite[i % self.elite_size]):
                    self.elite[i % self.elite_size] = x

            # Perform selection and crossover
            for i in range(self.swarm_size):
                r = np.random.rand()
                if r < 0.5:
                    self.swarm[i] = self.elite[np.random.randint(0, self.elite_size)]
                else:
                    self.swarm[i] = self.swarm[np.random.randint(0, self.swarm_size)]

            # Perform mutation
            for i in range(self.swarm_size):
                if np.random.rand() < 0.1:
                    self.swarm[i] += np.random.uniform(-0.1, 0.1, self.dim)

            # Evaluate function for each individual
            for i in range(self.swarm_size + self.elite_size):
                x = np.zeros(self.dim)
                if i < self.swarm_size:
                    x = self.swarm[i]
                else:
                    x = self.elite[i - self.swarm_size]
                f = func(x)
                if f < self.best:
                    self.best = f
                    self.elite[i - self.swarm_size] = x

            # Refine strategy with probability 0.15
            if np.random.rand() < self.refine_prob:
                for i in range(self.swarm_size):
                    # Select a random elite individual
                    r = np.random.randint(0, self.elite_size)
                    # Perform crossover with the selected elite individual
                    self.swarm[i] = (self.swarm[i] + self.elite[r]) / 2

        return self.best
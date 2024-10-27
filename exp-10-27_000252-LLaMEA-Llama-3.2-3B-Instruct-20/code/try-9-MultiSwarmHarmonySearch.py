import numpy as np
import random

class MultiSwarmHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.n_swarms = 5
        self.pbest = np.zeros((self.n_swarms, self.dim))
        self.best = np.inf
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.refine_prob = 0.2

    def __call__(self, func):
        for _ in range(self.budget):
            self.initialize_swarms()
            for _ in range(100):  # maximum number of iterations
                for i in range(self.n_swarms):
                    self.update_pbest(i)
                self.update_best()
                self.adaptation()
            self.evaluate_pbest(func)
        return self.best

    def initialize_swarms(self):
        self.swarms = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.n_swarms)]

    def update_pbest(self, i):
        if self.eval_func(self.swarms[i], func) < self.eval_func(self.pbest[i], func):
            self.pbest[i] = self.swarms[i]
            if random.random() < self.refine_prob:
                self.refine_pbest(i)

    def update_best(self):
        if self.eval_func(self.pbest[0], func) < self.best:
            self.best = self.eval_func(self.pbest[0], func)

    def adaptation(self):
        for i in range(self.n_swarms):
            if random.random() < self.mutation_rate:
                self.mutate(self.swarms[i])
            if random.random() < self.crossover_rate:
                self.crossover(self.swarms[i])

    def evaluate_pbest(self, func):
        for i in range(self.n_swarms):
            self.best = min(self.best, func(self.pbest[i]))

    def eval_func(self, x, func):
        return func(x)

    def mutate(self, x):
        idx = np.random.randint(0, self.dim)
        x[idx] += np.random.uniform(-1.0, 1.0)
        if x[idx] < -5.0:
            x[idx] = -5.0
        elif x[idx] > 5.0:
            x[idx] = 5.0

    def crossover(self, x):
        idx = np.random.randint(0, self.dim)
        x[idx] = (x[idx] + self.pbest[0][idx]) / 2
        if x[idx] < -5.0:
            x[idx] = -5.0
        elif x[idx] > 5.0:
            x[idx] = 5.0

    def refine_pbest(self, i):
        new_individual = self.swarms[i].copy()
        for _ in range(10):  # refine the individual 10 times
            new_individual = self.evaluate_fitness(new_individual)
        self.swarms[i] = new_individual

# Example usage:
def func(x):
    return sum(x**2)

ms = MultiSwarmHarmonySearch(100, 5)
best = ms(func)
print(best)
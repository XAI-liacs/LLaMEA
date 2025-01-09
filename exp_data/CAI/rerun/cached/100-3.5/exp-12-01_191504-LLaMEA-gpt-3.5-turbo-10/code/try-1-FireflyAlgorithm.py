import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.alpha = 0.2
        self.beta_min = 0.2
        self.gamma = 1.0

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        best = pop[np.argmin([func(x) for x in pop])]
        for _ in range(self.budget):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if func(pop[j]) < func(pop[i]):
                        r = np.linalg.norm(pop[i] - pop[j])
                        beta = self.beta_min * np.exp(-self.gamma * r ** 2)
                        step = self.alpha * (np.random.rand(self.dim) - 0.5) + beta * (pop[j] - pop[i])
                        pop[i] += step
            best = pop[np.argmin([func(x) for x in pop])]
        return best
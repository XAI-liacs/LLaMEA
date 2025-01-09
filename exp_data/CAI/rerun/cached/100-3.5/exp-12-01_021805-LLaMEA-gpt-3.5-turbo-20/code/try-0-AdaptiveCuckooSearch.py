import numpy as np

class AdaptiveCuckooSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.pa = 0.25
        self.nests = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_nest = self.nests[np.argmax([func(nest) for nest in self.nests])]

    def __call__(self, func):
        for _ in range(self.budget):
            new_nests = self.nests.copy()
            for i in range(self.population_size):
                step_size = 0.01 + 0.9 * (_ / self.budget)
                step = step_size * np.random.randn(self.dim)
                cuckoo = self.nests[i] + step
                cuckoo = np.clip(cuckoo, -5.0, 5.0)
                if func(cuckoo) < func(self.nests[i]) and np.random.rand() < self.pa:
                    new_nests[i] = cuckoo
            new_nests[np.argmax([func(nest) for nest in new_nests])] = self.best_nest
            self.nests = new_nests
            self.best_nest = self.nests[np.argmax([func(nest) for nest in self.nests])]
        return self.best_nest
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None
        self.sample_size = None
        self.sample_indices = None
        self.local_search = False
        self.best_func = None
        self.best_fitness = -np.inf
        self.p = 0.35

    def __call__(self, func):
        if self.func is None:
            self.func = func
            self.search_space = np.random.uniform(-5.0, 5.0, self.dim)
            self.sample_size = 1
            self.sample_indices = None

        if self.budget <= 0:
            raise ValueError("Budget is less than or equal to zero")

        for _ in range(self.budget):
            if self.sample_indices is None:
                self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
            else:
                self.sample_indices = np.random.choice(self.sample_indices, size=self.sample_size, replace=False)
            self.local_search = False

            if self.local_search:
                best_func = func(self.sample_indices)
                if np.abs(best_func - func(self.sample_indices)) < np.abs(self.best_func - self.best_fitness):
                    self.sample_indices = None
                    self.local_search = False
                    self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
                    self.sample_indices = self.sample_indices[:self.sample_size]
                else:
                    self.sample_indices = None
                    self.local_search = False

            if self.sample_indices is None:
                best_func = func(self.sample_indices)
                self.sample_indices = None
                self.local_search = False

            if np.abs(best_func - func(self.sample_indices)) < 1e-6:
                break

            if self.best_func is not None and np.abs(best_func - func(self.sample_indices)) < self.p * (self.best_fitness - np.abs(best_func - func(self.sample_indices))):
                self.best_func = best_func
                self.best_fitness = np.abs(best_func - func(self.sample_indices))
                self.p = 1 - self.p

        return func(self.sample_indices)
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
                if np.abs(best_func - func(self.sample_indices)) < np.abs(func(self.sample_indices) - func(self.sample_indices)):
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

        return func(self.sample_indices)

def adaptive_bbo(x, budget, dim, func):
    optimizer = BlackBoxOptimizer(budget, dim)
    return optimizer(x, func)

def adaptive_bbo_optimize(func, budget, dim, iterations=1000, tolerance=1e-6):
    best_func = None
    best_fitness = -np.inf
    for _ in range(iterations):
        x = np.random.uniform(-5.0, 5.0, dim)
        fitness = func(x)
        if fitness > best_fitness:
            best_func = func(x)
            best_fitness = fitness
        if np.abs(best_fitness - best_fitness) < tolerance:
            break
        new_x = adaptive_bbo(x, budget, dim, func)
        if best_func is None:
            best_func = func(new_x)
            best_fitness = fitness
        else:
            new_fitness = func(new_x)
            if new_fitness > best_fitness:
                best_func = func(new_x)
                best_fitness = new_fitness
    return best_func, best_fitness

# BBOB test suite
# Description: BBOB test suite for 24 noiseless functions
# Code: 
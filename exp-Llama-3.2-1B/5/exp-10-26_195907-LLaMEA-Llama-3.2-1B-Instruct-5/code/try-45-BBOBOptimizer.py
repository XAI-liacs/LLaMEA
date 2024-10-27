import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func, budget=100):
        while True:
            for _ in range(min(budget, self.budget // 0.05)):
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                if np.linalg.norm(func(x)) < self.budget // 2:
                    return x
            x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            self.search_space = np.vstack((self.search_space, x))
            self.search_space = np.delete(self.search_space, 0, axis=0)

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func, budget=100):
        while True:
            for _ in range(min(budget, self.budget // 0.05)):
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                if np.linalg.norm(func(x)) < self.budget // 2:
                    return x
            x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            self.search_space = np.vstack((self.search_space, x))
            self.search_space = np.delete(self.search_space, 0, axis=0)

class NovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.optimizer = NovelMetaheuristicOptimizer(budget, dim)
        self.logger = None

    def evaluate_fitness(self, individual, budget=100):
        if self.optimizer.logger is None:
            self.optimizer.logger = logging.getLogger()
            self.optimizer.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            self.optimizer.logger.addHandler(handler)
        return self.optimizer.func(individual)

    def __call__(self, func, budget=100):
        while True:
            fitness = self.evaluate_fitness(self.optimizer.func(np.random.uniform(-5.0, 5.0, size=(self.optimizer.dim, 2))), budget)
            if fitness < self.optimizer.func(np.random.uniform(-5.0, 5.0, size=(self.optimizer.dim, 2))):
                return np.random.uniform(-5.0, 5.0, size=(self.optimizer.dim, 2))
            x = self.optimizer.search_space[np.random.randint(0, self.optimizer.search_space.shape[0])]
            self.optimizer.search_space = np.vstack((self.optimizer.search_space, x))
            self.optimizer.search_space = np.delete(self.optimizer.search_space, 0, axis=0)

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
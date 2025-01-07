import numpy as np

class AdaptiveMemeticHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.harmony_memory_size = 10
        self.harmony_consideration_rate = 0.9
        self.pitch_adjustment_rate = 0.1
        self.local_search_rate = 0.2
        self.position = None
        self.pbest = None
        self.pbest_scores = None
        self.gbest = None
        self.gbest_score = float('inf')

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.pbest = np.copy(self.position)
        self.pbest_scores = np.full(self.population_size, float('inf'))
        self.harmony_memory = np.copy(self.position[:self.harmony_memory_size])

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i in range(self.population_size):
            if scores[i] < self.pbest_scores[i]:
                self.pbest_scores[i] = scores[i]
                self.pbest[i] = self.position[i]
            if scores[i] < self.gbest_score:
                self.gbest_score = scores[i]
                self.gbest = self.position[i]
        return scores

    def harmony_search(self):
        new_position = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            for d in range(self.dim):
                if np.random.rand() < self.harmony_consideration_rate:
                    new_position[i, d] = self.harmony_memory[np.random.randint(self.harmony_memory_size), d]
                    if np.random.rand() < self.pitch_adjustment_rate:
                        new_position[i, d] += 0.1 * np.random.randn()
                else:
                    new_position[i, d] = np.random.rand() * (self.harmony_memory[:, d].max() - self.harmony_memory[:, d].min()) + self.harmony_memory[:, d].min()
        self.position = new_position

    def local_search(self, position):
        return position + np.random.randn(*position.shape) * 0.01

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size

            if np.random.rand() < self.local_search_rate:
                for i in range(self.population_size):
                    self.position[i] = self.local_search(self.position[i])

            self.harmony_search()

        return self.gbest, self.gbest_score
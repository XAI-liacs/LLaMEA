import numpy as np

class AdaptiveMemoryMultiPhaseOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 60
        self.memory_size = 15  # Memory size for storing best solutions
        self.noise_amplitude = 0.05
        self.adaptive_phase_switch = 0.5  # Ratio to switch between exploration and exploitation
        self.exploration_factor = 0.9
        self.exploitation_factor = 0.1
        self.memory = None
        self.best_solution = None
        self.best_score = float('inf')

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.scores = np.full(self.population_size, float('inf'))
        self.memory = []

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.population])
        for i in range(self.population_size):
            if scores[i] < self.scores[i]:
                self.scores[i] = scores[i]
                if scores[i] < self.best_score:
                    self.best_score = scores[i]
                    self.best_solution = self.population[i]
        return scores

    def update_memory(self):
        sorted_indices = np.argsort(self.scores)
        for idx in sorted_indices[:self.memory_size]:
            if len(self.memory) < self.memory_size:
                self.memory.append(self.population[idx])
            else:
                worst_idx = np.argmax([func(p) for p in self.memory])
                if func(self.memory[worst_idx]) > self.scores[idx]:
                    self.memory[worst_idx] = self.population[idx]

    def adaptive_neighborhood_search(self, func):
        exploration = np.random.randn(self.population_size, self.dim) * self.exploration_factor
        exploitation = np.zeros((self.population_size, self.dim))
        for i, member in enumerate(self.memory):
            exploitation += (member - self.population) * self.exploitation_factor / len(self.memory)
        self.population += exploration + exploitation + np.random.randn(self.population_size, self.dim) * self.noise_amplitude

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        iteration = 0
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            self.update_memory()
            self.adaptive_neighborhood_search(func)
            iteration += 1

        return self.best_solution, self.best_score
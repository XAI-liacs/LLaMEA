import numpy as np

class QuantumInspiredBeeSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.employee_bees = self.population_size // 2
        self.onlooker_bees = self.population_size // 2
        self.scout_bees = 1
        self.adaptive_radius = 0.1
        self.food_sources = None
        self.food_source_fitness = None
        self.best_food_source = None
        self.best_fitness = float('inf')

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.food_sources = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.food_source_fitness = np.full(self.population_size, float('inf'))

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.food_sources])
        for i, score in enumerate(scores):
            if score < self.food_source_fitness[i]:
                self.food_source_fitness[i] = score
            if score < self.best_fitness:
                self.best_fitness = score
                self.best_food_source = self.food_sources[i]
        return scores

    def adaptive_exploration(self, position):
        offset = self.adaptive_radius * (np.random.rand(self.dim) - 0.5)
        return position + offset

    def employee_phase(self, func):
        for i in range(self.employee_bees):
            k = np.random.randint(0, self.employee_bees)
            while k == i:
                k = np.random.randint(0, self.employee_bees)
            candidate_solution = self.adaptive_exploration(self.food_sources[i])
            candidate_score = func(candidate_solution)
            if candidate_score < self.food_source_fitness[i]:
                self.food_sources[i] = candidate_solution
                self.food_source_fitness[i] = candidate_score

    def onlooker_phase(self, func):
        fitness_probabilities = self.food_source_fitness / np.sum(self.food_source_fitness)
        for _ in range(self.onlooker_bees):
            selected_index = np.random.choice(range(self.employee_bees), p=fitness_probabilities)
            candidate_solution = self.adaptive_exploration(self.food_sources[selected_index])
            candidate_score = func(candidate_solution)
            if candidate_score < self.food_source_fitness[selected_index]:
                self.food_sources[selected_index] = candidate_solution
                self.food_source_fitness[selected_index] = candidate_score

    def scout_phase(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        for i in range(self.scout_bees):
            self.food_sources[-(i + 1)] = lb + (ub - lb) * np.random.rand(self.dim)
            self.food_source_fitness[-(i + 1)] = float('inf')

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        while func_calls < self.budget:
            self.evaluate(func)
            self.employee_phase(func)
            self.onlooker_phase(func)
            self.scout_phase(func.bounds)
            func_calls += self.population_size

        return self.best_food_source, self.best_fitness
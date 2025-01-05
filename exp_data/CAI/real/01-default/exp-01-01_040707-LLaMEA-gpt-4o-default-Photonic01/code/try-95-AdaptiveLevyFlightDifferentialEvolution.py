import numpy as np

class AdaptiveLevyFlightDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
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

    def levy_flight(self):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(self.population_size, self.dim) * sigma
        v = np.random.randn(self.population_size, self.dim)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def quantum_inspired_position_update(self, current_position, best_position):
        phi = np.random.rand(*current_position.shape)
        return current_position + phi * (best_position - current_position)

    def update_population(self):
        new_population = np.copy(self.position)
        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = self.position[a] + self.mutation_factor * (self.position[b] - self.position[c])
            cross_points = np.random.rand(self.dim) < self.crossover_rate
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.position[i])
            new_population[i] = self.quantum_inspired_position_update(trial, self.gbest)
        return new_population

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            self.position = self.update_population()
            self.position += self.levy_flight()

        return self.gbest, self.gbest_score
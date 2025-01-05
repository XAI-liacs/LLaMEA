import numpy as np

class AdaptiveDifferentialEvolutionLevyFlight:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.position = None
        self.pbest = None
        self.gbest = None
        self.gbest_score = float('inf')

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        chaotic_sequence = np.random.rand(self.population_size, self.dim)
        self.position = lb + (ub - lb) * chaotic_sequence
        self.pbest = np.copy(self.position)

    def levy_flight(self, position):
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=position.shape)
        v = np.random.normal(0, 1, size=position.shape)
        step = u / np.abs(v) ** (1 / beta)
        return position + 0.01 * step

    def mutate(self, idx):
        idxs = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        return self.pbest[a] + self.F * (self.pbest[b] - self.pbest[c])

    def crossover(self, idx, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, self.position[idx])
        return trial

    def select(self, idx, trial, score):
        if score < self.gbest_score:
            self.gbest_score = score
            self.gbest = trial
        return trial if score < self.evaluate(self.position[idx]) else self.position[idx]

    def evaluate(self, position):
        return self.func(position)

    def __call__(self, func):
        func_calls = 0
        self.func = func
        self.initialize(func.bounds)

        while func_calls < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(i, mutant)
                trial = self.levy_flight(trial)
                score = self.evaluate(trial)
                self.position[i] = self.select(i, trial, score)
                func_calls += 1
                if func_calls >= self.budget:
                    break

        return self.gbest, self.gbest_score
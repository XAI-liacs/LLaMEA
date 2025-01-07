import numpy as np

class AdaptiveDEWithLevyFlight:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = 0.5  # Scaling factor
        self.CR = 0.9  # Crossover probability
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

    def levy_flight(self, step_size=0.1):
        u = np.random.normal(0, 1, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / 3))
        return step_size * step

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

    def mutate(self):
        indices = np.random.choice(self.population_size, 3, replace=False)
        a, b, c = self.position[indices[0]], self.position[indices[1]], self.position[indices[2]]
        return a + self.F * (b - c)

    def crossover(self, target, mutant):
        jrand = np.random.randint(self.dim)
        trial = np.array([mutant[j] if np.random.rand() < self.CR or j == jrand else target[j] for j in range(self.dim)])
        return trial

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        while func_calls < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate() + self.levy_flight()
                trial = self.crossover(self.position[i], mutant)
                trial_score = func(trial)
                func_calls += 1
                
                if trial_score < self.pbest_scores[i]:
                    self.pbest_scores[i] = trial_score
                    self.pbest[i] = trial
                    if trial_score < self.gbest_score:
                        self.gbest_score = trial_score
                        self.gbest = trial

                if func_calls >= self.budget:
                    break
        
        return self.gbest, self.gbest_score
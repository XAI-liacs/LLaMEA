import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.f = 0.8  # Scaling factor for DE
        self.cr = 0.9  # Crossover rate
        self.position = None
        self.velocity = None
        self.pbest = None
        self.pbest_scores = None
        self.gbest = None
        self.gbest_score = float('inf')
    
    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.velocity = np.random.rand(self.population_size, self.dim) - 0.5
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

    def update_velocity_position(self):
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        cognitive = self.c1 * r1 * (self.pbest - self.position)
        social = self.c2 * r2 * (self.gbest - self.position)
        
        self.velocity = self.w * self.velocity + cognitive + social
        self.position += self.velocity

    def differential_evolution_crossover(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        for i in range(self.population_size):
            idxs = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = self.position[idxs[0]], self.position[idxs[1]], self.position[idxs[2]]
            mutant = np.clip(a + self.f * (b - c), lb, ub)
            cross_points = np.random.rand(self.dim) < self.cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            self.position[i] = np.where(cross_points, mutant, self.position[i])

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            self.update_velocity_position()
            self.differential_evolution_crossover(func.bounds)

        return self.gbest, self.gbest_score
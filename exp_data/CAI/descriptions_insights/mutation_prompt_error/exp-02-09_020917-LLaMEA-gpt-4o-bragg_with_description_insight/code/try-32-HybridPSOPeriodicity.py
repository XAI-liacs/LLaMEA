import numpy as np
from scipy.optimize import minimize

class HybridPSOPeriodicity:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.population = None
        self.velocities = None
        self.lb = None
        self.ub = None
        self.pbest = None
        self.pbest_scores = None
        self.gbest = None
        self.gbest_score = np.inf
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5 + (0.3 * np.random.rand()) # Cognitive component with dynamic adjustment
        self.c2 = 1.5 # Social component

    def initialize_population(self, lb, ub, size):
        self.population = lb + (ub - lb) * np.random.rand(size, self.dim)
        self.velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (size, self.dim))
        self.pbest = np.copy(self.population)
        self.pbest_scores = np.full(size, np.inf)

    def periodic_constraint(self, position):
        period = (self.ub - self.lb) / self.dim
        period_position = self.lb + (np.round((position - self.lb) / period) * period)
        return np.clip(period_position, self.lb, self.ub)

    def particle_swarm_optimization(self, func):
        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                # Evaluate
                current_score = func(self.population[i])
                if current_score < self.pbest_scores[i]:
                    self.pbest[i] = self.population[i]
                    self.pbest_scores[i] = current_score
                if current_score < self.gbest_score:
                    self.gbest = self.population[i]
                    self.gbest_score = current_score

                # Update velocities and positions
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive = self.c1 * r1 * (self.pbest[i] - self.population[i])
                social = self.c2 * r2 * (self.gbest - self.population[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                self.population[i] = self.periodic_constraint(self.population[i] + self.velocities[i])

    def local_refinement(self, func):
        result = minimize(func, self.gbest, method='BFGS', bounds=[(self.lb[i], self.ub[i]) for i in range(self.dim)])
        if result.success:
            self.gbest = result.x
            self.gbest_score = func(result.x)

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(self.lb, self.ub, self.population_size)
        self.particle_swarm_optimization(func)
        self.local_refinement(func)
        return self.gbest
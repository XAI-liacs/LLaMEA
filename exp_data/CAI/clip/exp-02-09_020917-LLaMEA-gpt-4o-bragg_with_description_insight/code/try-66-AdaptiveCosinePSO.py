import numpy as np
from scipy.optimize import minimize

class AdaptiveCosinePSO:
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
        self.w = 0.7  # Initial inertia weight
        self.c1 = 1.8 # Cognitive component
        self.c2 = 1.2 # Social component

    def initialize_population(self, lb, ub, size):
        self.population = lb + (ub - lb) * np.random.rand(size, self.dim)
        self.velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (size, self.dim))
        self.pbest = np.copy(self.population)
        self.pbest_scores = np.full(size, np.inf)

    def periodicity_enforcement(self, position):
        # Use cosine similarity to encourage periodicity
        factor = np.cos(np.pi * (position - self.lb) / (self.ub - self.lb))
        return np.clip(self.lb + factor * (self.ub - self.lb), self.lb, self.ub)

    def adaptive_learning_rate(self, iteration):
        return 0.4 + 0.3 * (1 - np.cos(np.pi * iteration / self.budget))

    def particle_swarm_optimization(self, func):
        for iteration in range(self.budget - self.population_size):
            adaptive_lr = self.adaptive_learning_rate(iteration)
            for i in range(self.population_size):
                current_score = func(self.population[i])
                if current_score < self.pbest_scores[i]:
                    self.pbest[i] = self.population[i]
                    self.pbest_scores[i] = current_score
                if current_score < self.gbest_score:
                    self.gbest = self.population[i]
                    self.gbest_score = current_score

                r1, r2 = np.random.rand(), np.random.rand()
                cognitive = self.c1 * r1 * (self.pbest[i] - self.population[i])
                social = self.c2 * r2 * (self.gbest - self.population[i])
                velocity_update = (self.w * self.velocities[i] + cognitive + social)
                self.velocities[i] = adaptive_lr * velocity_update

                vmax = 0.2 * (self.ub - self.lb)
                self.velocities[i] = np.clip(self.velocities[i], -vmax, vmax)

                self.population[i] = self.periodicity_enforcement(self.population[i] + self.velocities[i])

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
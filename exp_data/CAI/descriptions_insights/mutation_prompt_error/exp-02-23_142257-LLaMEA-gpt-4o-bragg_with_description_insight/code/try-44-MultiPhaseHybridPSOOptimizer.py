import numpy as np
from scipy.optimize import minimize

class MultiPhaseHybridPSOOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 10 * dim  # Swarm size to balance exploration and convergence
        self.population = None
        self.velocities = None
        self.best_solution = None
        self.best_score = float('-inf')
        self.eval_count = 0
        self.personal_best_positions = None
        self.personal_best_scores = None

    def initialize_swarm(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.best_solution = self.population[0]
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.swarm_size, float('-inf'))

    def update_velocities(self, global_best, w=0.5, c1=1.5, c2=1.5):
        r1 = np.random.rand(self.swarm_size, self.dim)
        r2 = np.random.rand(self.swarm_size, self.dim)
        cognitive_component = c1 * r1 * (self.personal_best_positions - self.population)
        social_component = c2 * r2 * (global_best - self.population)
        self.velocities = w * self.velocities + cognitive_component + social_component

    def apply_periodicity_penalty(self, x):
        period = self.dim // 2
        periodic_deviation = np.sum((x[:period] - x[period:2*period]) ** 2)
        deviation_penalty = np.sum((x - np.roll(x, period)) ** 2)
        return periodic_deviation + deviation_penalty

    def local_refinement(self, func, lb, ub):
        result = minimize(func, self.best_solution, bounds=np.c_[lb, ub], method='L-BFGS-B')
        self.eval_count += result.nfev
        if result.fun > self.best_score:
            self.best_score = result.fun
            self.best_solution = result.x

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_swarm(lb, ub)

        while self.eval_count < self.budget:
            for i in range(self.swarm_size):
                if self.eval_count >= self.budget:
                    break

                self.population[i] = np.clip(self.population[i] + self.velocities[i], lb, ub)
                score = func(self.population[i]) - self.apply_periodicity_penalty(self.population[i])
                self.eval_count += 1

                if score > self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.population[i]

                if score > self.best_score:
                    self.best_score = score
                    self.best_solution = self.population[i]

            global_best = self.population[np.argmax(self.personal_best_scores)]
            self.update_velocities(global_best)

            if self.eval_count < self.budget / 2:
                self.local_refinement(func, lb, ub)

        return self.best_solution
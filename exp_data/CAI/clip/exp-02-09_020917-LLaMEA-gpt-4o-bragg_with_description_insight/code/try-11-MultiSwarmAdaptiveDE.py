import numpy as np
from scipy.optimize import minimize

class MultiSwarmAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarms_count = 3
        self.swarm_size = 10 * dim
        self.total_population_size = self.swarm_size * self.swarms_count
        self.populations = [None] * self.swarms_count
        self.velocities = [None] * self.swarms_count
        self.pbest = [None] * self.swarms_count
        self.pbest_scores = [None] * self.swarms_count
        self.gbest = None
        self.gbest_score = np.inf
        self.lb, self.ub = None, None
        self.w = 0.5
        self.c1 = 1.7
        self.c2 = 1.5
        self.f = 0.8  # DE crossover factor
        self.cr = 0.9  # DE crossover rate

    def initialize_population(self, lb, ub, size):
        for s in range(self.swarms_count):
            self.populations[s] = lb + (ub - lb) * np.random.rand(size, self.dim)
            self.velocities[s] = np.random.uniform(-abs(ub - lb), abs(ub - lb), (size, self.dim))
            self.pbest[s] = np.copy(self.populations[s])
            self.pbest_scores[s] = np.full(size, np.inf)

    def periodic_constraint(self, position, period_factor):
        period = period_factor * (self.ub - self.lb) / self.dim
        period_position = self.lb + (np.round((position - self.lb) / period) * period)
        return np.clip(period_position, self.lb, self.ub)

    def multi_swarm_optimization(self, func):
        evals_per_swarm = (self.budget - self.total_population_size) // self.swarms_count
        for _ in range(evals_per_swarm):
            for s in range(self.swarms_count):
                period_factor = 1 + 0.1 * s  # Different periodicity for each swarm
                for i in range(self.swarm_size):
                    current_score = func(self.populations[s][i])
                    if current_score < self.pbest_scores[s][i]:
                        self.pbest[s][i] = self.populations[s][i]
                        self.pbest_scores[s][i] = current_score
                    if current_score < self.gbest_score:
                        self.gbest = self.populations[s][i]
                        self.gbest_score = current_score

                    # Update velocities and positions using PSO
                    r1, r2 = np.random.rand(), np.random.rand()
                    cognitive = self.c1 * r1 * (self.pbest[s][i] - self.populations[s][i])
                    social = self.c2 * r2 * (self.gbest - self.populations[s][i])
                    self.velocities[s][i] = self.w * self.velocities[s][i] + cognitive + social
                    self.populations[s][i] = self.periodic_constraint(self.populations[s][i] + self.velocities[s][i], period_factor)

                    # Apply DE crossover
                    a, b, c = self.populations[s][np.random.randint(0, self.swarm_size, 3)]
                    mutant = np.clip(a + self.f * (b - c), self.lb, self.ub)
                    cross_points = np.random.rand(self.dim) < self.cr
                    trial = np.where(cross_points, mutant, self.populations[s][i])
                    trial = self.periodic_constraint(trial, period_factor)
                    trial_score = func(trial)
                    if trial_score < current_score:
                        self.populations[s][i] = trial

    def local_refinement(self, func):
        result = minimize(func, self.gbest, method='BFGS', bounds=[(self.lb[i], self.ub[i]) for i in range(self.dim)])
        if result.success:
            self.gbest = result.x
            self.gbest_score = func(result.x)

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(self.lb, self.ub, self.swarm_size)
        self.multi_swarm_optimization(func)
        self.local_refinement(func)
        return self.gbest
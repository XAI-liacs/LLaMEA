import numpy as np
from scipy.optimize import minimize

class PeriodicDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.CR = 0.9  # Crossover probability
        self.F = 0.8   # Differential weight
        self.population = None
        self.bounds = None
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.rand(self.pop_size, self.dim) * (ub - lb) + lb
        # Encourage periodicity by initializing with sinusoidal patterns
        for i in range(self.pop_size):
            period = np.random.randint(1, self.dim // 2)
            self.population[i] = np.sin(np.linspace(0, 2 * np.pi * period, self.dim)) * (ub - lb) / 2 + (lb + ub) / 2

    def evaluate(self, func, x):
        if self.evaluations >= self.budget:
            raise Exception("Budget exhausted")
        self.evaluations += 1
        return func(x)

    def mutate(self, target_idx):
        indices = list(range(self.pop_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def local_optimization(self, best_individual, func):
        res = minimize(func, best_individual, method='L-BFGS-B', bounds=list(zip(self.bounds.lb, self.bounds.ub)))
        return res.x

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population(self.bounds.lb, self.bounds.ub)
        best_solution = None
        best_score = float('inf')

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                target = self.population[i]
                mutant = self.mutate(i)
                trial = self.crossover(target, mutant)
                trial_score = self.evaluate(func, trial)
                target_score = self.evaluate(func, target)

                if trial_score < target_score:
                    self.population[i] = trial
                    if trial_score < best_score:
                        best_score = trial_score
                        best_solution = trial

                if self.evaluations >= self.budget:
                    break

        # Fine-tune the best solution found
        if best_solution is not None:
            best_solution = self.local_optimization(best_solution, func)

        return best_solution
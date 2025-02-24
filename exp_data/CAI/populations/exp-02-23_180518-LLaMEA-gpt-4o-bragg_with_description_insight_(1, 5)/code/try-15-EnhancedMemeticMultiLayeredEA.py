import numpy as np
from scipy.optimize import minimize

class EnhancedMemeticMultiLayeredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.pop = None
        self.velocities = None
        self.p_best = None
        self.p_best_val = None
        self.g_best = None
        self.g_best_val = np.inf
        self.evaluations = 0
        self.f = 0.8  # DE scaling factor
        self.cr = 0.9 # DE crossover probability

    def initialize_population(self, lb, ub):
        self.pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.p_best = self.pop.copy()
        self.p_best_val = np.full(self.population_size, np.inf)

    def promote_periodicity(self, solution):
        periodic_solution = solution.copy()
        for i in range(self.dim):
            periodic_solution[i] = (solution[i] + solution[(i + 1) % self.dim]) / 2
        return periodic_solution

    def differential_evolution(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        
        fitness = np.apply_along_axis(func, 1, self.pop)
        self.evaluations += self.population_size
        self.p_best_val = np.minimum(self.p_best_val, fitness)
        better_mask = fitness < self.p_best_val
        self.p_best[better_mask] = self.pop[better_mask]
        
        if fitness.min() < self.g_best_val:
            self.g_best_val = fitness.min()
            self.g_best = self.pop[fitness.argmin()].copy()

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                candidates = [index for index in range(self.population_size) if index != i]
                a, b, c = self.pop[np.random.choice(candidates, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.pop[i])
                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    self.pop[i] = trial
                    if trial_fitness < self.g_best_val:
                        self.g_best_val = trial_fitness
                        self.g_best = trial

    def local_pattern_search(self, func):
        if self.g_best is not None:
            result = minimize(func, self.promote_periodicity(self.g_best), method='L-BFGS-B', bounds=[(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)])
            self.evaluations += result.nfev

            if result.fun < self.g_best_val:
                self.g_best_val = result.fun
                self.g_best = result.x

    def __call__(self, func):
        self.differential_evolution(func)
        self.local_pattern_search(func)
        return self.g_best, self.g_best_val
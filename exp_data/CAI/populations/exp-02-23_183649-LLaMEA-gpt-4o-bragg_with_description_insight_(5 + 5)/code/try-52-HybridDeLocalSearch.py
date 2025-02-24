import numpy as np
import scipy.optimize as opt

class HybridDeLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 5 * dim
        self.F = np.random.uniform(0.5, 0.9)
        self.CR = np.random.uniform(0.5, 0.9)
        self.pop = None
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        quasi_opposite_pop = lb + ub - self.pop
        self.pop = np.vstack((self.pop, quasi_opposite_pop))

    def evaluate_population(self, func):
        scores = np.apply_along_axis(func, 1, self.pop)
        self.evaluations += len(scores)
        return scores

    def update_rates(self):
        progress_ratio = self.evaluations / self.budget
        self.F = 0.5 + progress_ratio * 0.4  # Gradually increase F
        self.CR = 0.9 - progress_ratio * 0.4  # Gradually decrease CR

    def differential_evolution_step(self, func, scores, lb, ub):
        self.update_rates()  # Update rates based on progress
        for i in range(min(self.population_size, self.budget - self.evaluations)):
            if self.evaluations >= self.budget:
                break
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + self.F * (b - c)
            mutant = np.clip(mutant, lb, ub)
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.pop[i])
            f_trial = func(trial)
            self.evaluations += 1
            if f_trial < scores[i]:
                scores[i] = f_trial
                self.pop[i] = trial
                if f_trial < self.best_score:
                    self.best_score = f_trial
                    self.best_solution = trial

    def local_search(self, func, lb, ub):
        if self.best_solution is not None:
            result = opt.minimize(func, self.best_solution, bounds=list(zip(lb, ub)), method='L-BFGS-B')
            if result.fun < self.best_score:
                self.best_score = result.fun
                self.best_solution = result.x
                self.evaluations += result.nfev

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        scores = self.evaluate_population(func)
        while self.evaluations < self.budget:
            self.differential_evolution_step(func, scores, lb, ub)
            self.pop[np.argmin(scores)] = self.best_solution
            if self.evaluations < self.budget:
                self.local_search(func, lb, ub)
        return self.best_solution
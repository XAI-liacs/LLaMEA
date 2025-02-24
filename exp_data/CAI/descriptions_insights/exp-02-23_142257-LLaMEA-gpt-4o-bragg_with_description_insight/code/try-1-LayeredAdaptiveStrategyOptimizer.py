import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

class LayeredAdaptiveStrategyOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.population = None
        self.best_solution = None
        self.best_score = float('-inf')
        self.eval_count = 0
        self.kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(kernel=self.kernel)

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.best_solution = self.population[0]

    def reflectivity_score(self, x):
        period = int(self.dim / 2)
        periodic_deviation = np.sum((x[:period] - x[period:2*period]) ** 2)
        return periodic_deviation

    def differential_evolution(self, func, lb, ub):
        F_initial = 0.5
        CR = 0.9
        
        for i in range(self.population_size):
            if self.eval_count >= self.budget:
                break

            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            F_adaptive = F_initial * np.exp(-i / self.population_size)  # Adaptive F
            mutant = np.clip(a + F_adaptive * (b - c), lb, ub)

            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True

            trial = np.where(cross_points, mutant, self.population[i])

            trial_score = func(trial)
            self.eval_count += 1

            if trial_score > func(self.population[i]) - self.reflectivity_score(trial):
                self.population[i] = trial

            if trial_score > self.best_score:
                self.best_score = trial_score
                self.best_solution = trial

    def bayesian_focus(self, func, lb, ub):
        if self.eval_count < self.budget:
            X = self.population
            y = -np.array([func(x) for x in X])  # Negative for maximization
            self.eval_count += len(X)
            self.gp.fit(X, y)

            bounds = np.array([lb, ub]).T
            res = minimize(lambda x: -self.gp.predict(x.reshape(1, -1))[0], self.best_solution, bounds=bounds, method='L-BFGS-B')
            if res.fun > -self.best_score:
                self.best_score = -res.fun
                self.best_solution = res.x

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)

        while self.eval_count < self.budget:
            self.differential_evolution(func, lb, ub)
            self.bayesian_focus(func, lb, ub)

        return self.best_solution
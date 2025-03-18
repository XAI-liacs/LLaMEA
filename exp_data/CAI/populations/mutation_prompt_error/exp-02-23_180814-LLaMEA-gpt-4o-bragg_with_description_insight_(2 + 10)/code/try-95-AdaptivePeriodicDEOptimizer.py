import numpy as np
from scipy.optimize import minimize

class AdaptivePeriodicDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim 
        self.f_min = 0.5
        self.f_max = 0.9
        self.cr_min = 0.1
        self.cr_max = 0.9
        self.population = None
        self.best_solution = None
        self.best_score = np.inf
        self.bounds = None

    def _initialize_population(self):
        lower_bound, upper_bound = self.bounds
        self.population = np.random.rand(self.population_size, self.dim) * (upper_bound - lower_bound) + lower_bound
        period = self.dim // 2
        for i in range(self.population_size):
            for j in range(0, self.dim, period):
                self.population[i, j:j+period] = np.sin(self.population[i, :period])  # Modified periodicity with sine

    def _evaluate(self, func):
        scores = np.array([func(ind) for ind in self.population])
        best_index = np.argmin(scores)
        if scores[best_index] < self.best_score:
            self.best_score = scores[best_index]
            self.best_solution = self.population[best_index]
        return scores

    def _mutate(self, target_idx, scores):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        f_adaptive = self.f_min + (self.f_max - self.f_min) * np.std(scores) / (np.mean(scores) + 1e-9)
        best_indices = np.argsort(scores)[:3]
        a, b, c = np.random.choice(best_indices, 3, replace=True)
        mutant = self.population[a] + f_adaptive * (self.population[b] - self.population[c])
        period = self.dim // 2
        for j in range(0, self.dim, period):
            mutant[j:j+period] = np.mean(mutant[:period]) + 0.01 * np.random.rand()  # Added small randomness
        return np.clip(mutant, self.bounds[0], self.bounds[1])

    def _crossover(self, target, mutant, adaptive_cr):
        crossover_mask = np.random.rand(self.dim) < adaptive_cr + 0.2  # Further increased crossover probability
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def _local_optimize(self, x0, func):
        result = minimize(func, x0, method='L-BFGS-B', bounds=[(lb, ub) for lb, ub in zip(*self.bounds)])
        return result.x

    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        self._initialize_population()

        num_evaluations = 0
        while num_evaluations < self.budget:
            scores = self._evaluate(func)
            self.population_size = max(5, int(self.population_size * (1 - num_evaluations / self.budget)))
            num_evaluations += self.population_size

            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                mutant = self._mutate(i, scores)
                adaptive_cr = self.cr_min + (self.cr_max - self.cr_min) * (scores[i] - self.best_score) / (np.max(scores) - np.min(scores) + 1e-9)
                trial = self._crossover(self.population[i], mutant, adaptive_cr)
                trial_score = func(trial)
                num_evaluations += 1

                if trial_score < scores[i]:
                    self.population[i] = trial
                    scores[i] = trial_score
                    if trial_score < self.best_score:
                        self.best_score = trial_score
                        self.best_solution = trial

            if num_evaluations < self.budget:
                local_best = self._local_optimize(self.best_solution, func)
                local_best_score = func(local_best)
                num_evaluations += 1
                if local_best_score < self.best_score:
                    self.best_score = local_best_score
                    self.best_solution = local_best

        return self.best_solution
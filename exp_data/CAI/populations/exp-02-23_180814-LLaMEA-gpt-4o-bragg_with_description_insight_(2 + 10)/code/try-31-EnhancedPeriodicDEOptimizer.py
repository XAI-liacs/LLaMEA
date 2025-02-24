import numpy as np
from scipy.optimize import minimize

class EnhancedPeriodicDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim 
        self.f = 0.8  # Initial DE scaling factor
        self.cr = 0.9 # Crossover probability
        self.population = None
        self.best_solution = None
        self.best_score = np.inf
        self.bounds = None

    def _initialize_population(self):
        # Initialize population with periodic encouragement
        lower_bound, upper_bound = self.bounds
        self.population = np.random.rand(self.population_size, self.dim) * (upper_bound - lower_bound) + lower_bound
        # Encouraging periodicity in initial population
        period = self.dim // 2
        for i in range(self.population_size):
            for j in range(0, self.dim, period):
                self.population[i, j:j+period] = self.population[i, :period]

    def _evaluate(self, func):
        scores = np.array([func(ind) for ind in self.population])
        best_index = np.argmin(scores)
        if scores[best_index] < self.best_score:
            self.best_score = scores[best_index]
            self.best_solution = self.population[best_index]
        return scores

    def _mutate(self, target_idx):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        diversity_factor = np.std(self.population, axis=0)
        adaptive_f = self.f * (1 + 0.5 * np.random.randn())
        mutant = self.population[a] + adaptive_f * (self.population[b] - self.population[c]) * diversity_factor
        return np.clip(mutant, self.bounds[0], self.bounds[1])

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.cr
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def _local_optimize(self, x0, func):
        result = minimize(func, x0, method='L-BFGS-B', bounds=[(lb, ub) for lb, ub in zip(*self.bounds)])
        return result.x

    def _periodicity_aware_refine(self, solution, func):
        # Small gradient descent step to encourage periodic solutions
        period = self.dim // 2
        for j in range(0, self.dim, period):
            perturbation = np.random.randn(period) * 0.01
            solution[j:j+period] += perturbation
            solution[j:j+period] = np.clip(solution[j:j+period], self.bounds[0][j:j+period], self.bounds[1][j:j+period])
        return solution

    def __call__(self, func):
        # Set bounds for convenience
        self.bounds = (func.bounds.lb, func.bounds.ub)
        self._initialize_population()

        num_evaluations = 0
        while num_evaluations < self.budget:
            scores = self._evaluate(func)
            num_evaluations += self.population_size
            self.cr = 0.9 - 0.5 * (num_evaluations / self.budget)  # Adaptive crossover probability

            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                mutant = self._mutate(i)
                trial = self._crossover(self.population[i], mutant)
                trial_score = func(trial)
                num_evaluations += 1

                if trial_score < scores[i]:
                    self.population[i] = trial
                    scores[i] = trial_score
                    if trial_score < self.best_score:
                        self.best_score = trial_score
                        self.best_solution = trial
            
            if num_evaluations < self.budget:
                # Perform periodicity-aware local optimization
                local_best = self._local_optimize(self.best_solution, func)
                local_best = self._periodicity_aware_refine(local_best, func)
                local_best_score = func(local_best)
                num_evaluations += 1
                if local_best_score < self.best_score:
                    self.best_score = local_best_score
                    self.best_solution = local_best

        return self.best_solution
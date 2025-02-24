import numpy as np
from scipy.optimize import minimize

class RefinedHybridDEBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.current_budget = 0

    def _initialize_population(self, bounds):
        pop = np.random.rand(self.population_size, self.dim)
        pop = bounds.lb + (bounds.ub - bounds.lb) * pop
        return pop

    def _select_best(self, population, scores):
        best_idx = np.argmin(scores)
        return population[best_idx], scores[best_idx]

    def _mutate(self, target_idx, population):
        selected_indices = np.random.choice(np.delete(np.arange(self.population_size), target_idx), 3, replace=False)
        a, b, c = population[selected_indices]
        period = self.dim // 2
        phase_shift = np.random.randint(0, period)
        mutant = a + self.mutation_factor * (b - c)
        mutant = np.roll(mutant, phase_shift)  # Apply phase shift
        return mutant

    def _crossover(self, target, mutant, bounds):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return np.clip(trial, bounds.lb, bounds.ub)

    def _evaluate_population(self, func, population):
        scores = np.array([func(ind) for ind in population])
        self.current_budget += len(population)
        return scores

    def _local_optimization(self, best, func, bounds):
        result = minimize(func, best, method='L-BFGS-B', bounds=list(zip(bounds.lb, bounds.ub)))
        return result.x if result.success else best

    def _enforce_periodicity(self, solution):
        period = self.dim // 2
        for i in range(period):
            solution[i + period] = solution[i]
        return solution

    def __call__(self, func):
        bounds = func.bounds
        population = self._initialize_population(bounds)
        scores = self._evaluate_population(func, population)

        while self.current_budget < self.budget:
            for i in range(self.population_size):
                if self.current_budget >= self.budget:
                    break
                target = population[i]
                mutant = self._mutate(i, population)
                self.crossover_rate = 0.5 + 0.2 * np.random.rand()
                trial = self._crossover(target, mutant, bounds)
                trial = self._enforce_periodicity(trial)
                trial_score = func(trial)
                self.current_budget += 1
                if trial_score < scores[i]:
                    population[i] = trial
                    scores[i] = trial_score

            best, best_score = self._select_best(population, scores)
            best = self._local_optimization(best, func, bounds)
            best = self._enforce_periodicity(best)

        return best
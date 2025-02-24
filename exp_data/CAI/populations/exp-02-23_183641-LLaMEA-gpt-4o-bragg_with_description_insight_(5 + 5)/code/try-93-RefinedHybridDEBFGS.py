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
        adaptive_mutation_factor = np.random.uniform(0.65, 0.9) * (1 - self.current_budget / self.budget)  # Adjusted mutation factor
        mutant = a + adaptive_mutation_factor * (b - c)
        return mutant

    def _dynamic_crossover(self, target, mutant, bounds):
        dynamic_crossover_rate = self.crossover_rate * (1 - self.current_budget / self.budget)
        cross_points = np.random.rand(self.dim) < dynamic_crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return np.clip(trial, bounds.lb, bounds.ub)

    def _evaluate_population(self, func, population):
        scores = np.array([func(ind) for ind in population])
        self.current_budget += len(population)
        return scores

    def _local_optimization(self, best, func, bounds):
        result = minimize(func, best, method='L-BFGS-B', bounds=list(zip(bounds.lb, bounds.ub)), options={'maxiter': 100})
        return result.x if result.success else best

    def _adaptive_periodicity(self, solution):
        period = max(1, self.dim // 5)
        solution[:period] = np.mean(solution[:period])  # Average initial block
        for i in range(period):
            solution[i::period] = solution[i]
        return solution

    def __call__(self, func):
        bounds = func.bounds
        population = self._initialize_population(bounds)
        scores = self._evaluate_population(func, population)
        best, best_score = self._select_best(population, scores)

        while self.current_budget < self.budget:
            for i in range(self.population_size):
                if self.current_budget >= self.budget:
                    break
                target = population[i]
                mutant = self._mutate(i, population)
                trial = self._dynamic_crossover(target, mutant, bounds)
                trial = self._adaptive_periodicity(trial)
                trial_score = func(trial)
                self.current_budget += 1
                if trial_score < scores[i]:
                    population[i] = trial
                    scores[i] = trial_score
                    elitism_factor = 1 - (self.current_budget / self.budget)  # Adaptive elitism factor
                    if trial_score < best_score * elitism_factor:  # Introduced adaptive elitism
                        best, best_score = trial, trial_score

            best = self._local_optimization(best, func, bounds)
            best = self._adaptive_periodicity(best)

        return best
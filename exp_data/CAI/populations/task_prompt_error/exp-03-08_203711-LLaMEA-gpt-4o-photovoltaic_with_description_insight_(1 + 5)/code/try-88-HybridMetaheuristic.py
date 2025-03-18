import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.f = 0.8
        self.cr = 0.9
        self.history = []

    def _de_mutation(self, pop):
        idxs = np.random.choice(range(self.population_size), 3, replace=False)
        a, b, c = pop[idxs]
        adapt_f = self.f * (1 - (len(self.history) / self.budget))
        return np.clip(a + adapt_f * (b - c), 0, 1)

    def _de_crossover(self, target, mutant):
        adapt_cr = max(self.cr * (0.5 + 0.5 * np.random.rand()), 0.6)
        cross_points = np.random.rand(self.dim) < adapt_cr
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        return np.where(cross_points, mutant, target)

    def _local_search(self, x, bounds, func):
        def wrapped_func(x_in):
            self.history.append(x_in)
            return func(x_in)
        
        result = minimize(wrapped_func, x, bounds=bounds, method='L-BFGS-B')
        return result.x

    def _adaptive_layer_growth(self, base_dim, current_dim):
        return min(current_dim + 4, self.dim)
    
    def _adaptive_population_scaling(self, evaluations):  # New method
        return max(10, int(self.population_size * (1 - evaluations / self.budget)))
    
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        best_solution = None
        best_value = float('inf')
        current_dim = 10

        pop = np.random.rand(self.population_size, current_dim)
        pop = bounds[:, 0] + pop * (bounds[:, 1] - bounds[:, 0])

        evaluations = 0
        while evaluations < self.budget:
            new_pop = np.zeros_like(pop)

            for i in range(self.population_size):
                target = pop[i]
                mutant = self._de_mutation(pop)
                trial = self._de_crossover(target, mutant)
                trial = self._local_search(trial, bounds, func)

                f_trial = func(trial) + 0.1 * np.var(trial)
                f_target = func(target) + 0.1 * np.var(target)

                if f_trial < f_target:
                    new_pop[i] = trial
                else:
                    new_pop[i] = target

                evaluations += 1
                if evaluations >= self.budget:
                    break

            pop = new_pop

            for individual in pop:
                value = func(individual) + 0.1 * np.var(individual)
                if value < best_value:
                    best_value = value
                    best_solution = individual

            current_dim = self._adaptive_layer_growth(current_dim, current_dim)
            pop = np.hstack((pop, np.random.rand(self.population_size, current_dim - pop.shape[1])))
            self.population_size = self._adaptive_population_scaling(evaluations)  # Adjust population size

        return best_solution
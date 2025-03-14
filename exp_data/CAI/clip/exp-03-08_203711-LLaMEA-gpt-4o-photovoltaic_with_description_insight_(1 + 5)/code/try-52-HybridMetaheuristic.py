import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.f = 0.8  # DE scale factor
        self.cr = 0.9  # DE crossover probability
        self.history = []
        self.num_populations = 3  # Multi-population strategy

    def _de_mutation(self, pop):
        idxs = np.random.choice(range(self.population_size), 3, replace=False)
        a, b, c = pop[idxs]
        adapt_f = self.f * (1 - (len(self.history) / self.budget))  # Adaptive scaling
        return np.clip(a + adapt_f * (b - c), 0, 1)

    def _de_crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.cr
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        return np.where(cross_points, mutant, target)

    def _local_search(self, x, bounds, func):
        def wrapped_func(x_in):
            self.history.append(x_in)
            return func(x_in)
        
        result = minimize(wrapped_func, x, bounds=bounds, method='L-BFGS-B')
        return result.x

    def _adaptive_layer_growth(self, current_dim):
        return min(current_dim + 2, self.dim)

    def _multi_population_update(self, populations, best_solution):
        new_populations = []
        for pop in populations:
            if np.random.rand() < 0.5:
                new_populations.append(np.copy(pop))
            else:
                new_populations.append(np.copy(best_solution))
        return new_populations

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        best_solution = None
        best_value = float('inf')
        current_dim = 10  # Start with 10 layers

        populations = [np.random.rand(self.population_size, current_dim) for _ in range(self.num_populations)]
        populations = [bounds[:, 0] + pop * (bounds[:, 1] - bounds[:, 0]) for pop in populations]

        evaluations = 0
        while evaluations < self.budget:
            new_populations = []

            for pop in populations:
                new_pop = np.zeros_like(pop)
                for i in range(self.population_size):
                    target = pop[i]
                    mutant = self._de_mutation(pop)
                    trial = self._de_crossover(target, mutant)
                    trial = self._local_search(trial, bounds, func)

                    f_trial = func(trial)
                    f_target = func(target)

                    if f_trial < f_target:
                        new_pop[i] = trial
                    else:
                        new_pop[i] = target

                    evaluations += 1
                    if evaluations >= self.budget:
                        break

                new_populations.append(new_pop)

            populations = self._multi_population_update(new_populations, best_solution)

            for pop in populations:
                for individual in pop:
                    value = func(individual)
                    if value < best_value:
                        best_value = value
                        best_solution = individual

            current_dim = self._adaptive_layer_growth(current_dim)
            for pop in populations:
                extra_layers = np.random.rand(self.population_size, current_dim - pop.shape[1])
                pop = np.hstack((pop, extra_layers))

        return best_solution
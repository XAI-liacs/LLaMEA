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

    def _de_mutation(self, pop):
        idxs = np.random.choice(range(self.population_size), 3, replace=False)
        a, b, c = pop[idxs]
        adapt_f = self.f * (1 - (len(self.history) / self.budget))
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

    def _adaptive_layer_growth(self, base_dim, current_dim):
        return min(current_dim + 2, self.dim)
    
    def _adaptive_population_size(self, evaluations):
        return max(10, int(30 * (1 - evaluations / self.budget)))

    def _perturbation(self, individual):
        perturbation_scale = 0.05
        perturbation = np.random.uniform(-perturbation_scale, perturbation_scale, size=individual.shape)
        return individual + perturbation
    
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        best_solution = None
        best_value = float('inf')
        current_dim = 10

        evaluations = 0
        while evaluations < self.budget:
            self.population_size = self._adaptive_population_size(evaluations)
            pop = np.random.rand(self.population_size, current_dim)
            pop = bounds[:, 0] + pop * (bounds[:, 1] - bounds[:, 0])

            new_pop = np.zeros_like(pop)

            for i in range(self.population_size):
                target = pop[i]
                mutant = self._de_mutation(pop)
                trial = self._de_crossover(target, mutant)
                trial = self._local_search(trial, bounds, func)
                trial = self._perturbation(trial)

                f_trial = func(trial)
                f_target = func(target)

                if f_trial < f_target:
                    new_pop[i] = trial
                else:
                    new_pop[i] = target

                evaluations += 1
                if evaluations >= self.budget:
                    break

            pop = new_pop

            for individual in pop:
                value = func(individual)
                if value < best_value:
                    best_value = value
                    best_solution = individual

            current_dim = self._adaptive_layer_growth(current_dim, current_dim)
            pop = np.hstack((pop, np.random.rand(self.population_size, current_dim - pop.shape[1])))

        return best_solution
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
        self.layer_roles = np.random.randint(0, 2, size=self.dim)  # Added modular role detection
        self.robustness_weight = 0.05  # Added robustness metric weight

    def _de_mutation(self, pop):
        idxs = np.random.choice(range(self.population_size), 3, replace=False)
        a, b, c = pop[idxs]
        return np.clip(a + self.f * (b - c), 0, 1)

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

    def _robustness_penalty(self, x):  # Added robustness penalty calculation
        perturbation = np.random.uniform(-0.01, 0.01, size=x.shape)  
        return self.robustness_weight * np.linalg.norm(perturbation)

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

                f_trial = func(trial) + self._robustness_penalty(trial)  # Included robustness
                f_target = func(target) + self._robustness_penalty(target)  # Included robustness

                if f_trial < f_target:
                    new_pop[i] = trial
                else:
                    new_pop[i] = target

                evaluations += 1
                if evaluations >= self.budget:
                    break

            pop = new_pop

            for individual in pop:
                value = func(individual) + self._robustness_penalty(individual)  # Included robustness
                if value < best_value:
                    best_value = value
                    best_solution = individual

            current_dim = self._adaptive_layer_growth(current_dim, current_dim)
            pop = np.hstack((pop, np.random.rand(self.population_size, current_dim - pop.shape[1])))

        return best_solution
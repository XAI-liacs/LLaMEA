import numpy as np
from scipy.optimize import minimize

class HybridOptimizationEnhancedRefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_de = int(budget * 0.60) # Increased DE budget for better exploration
        self.num_nm = budget - self.num_de

    def __call__(self, func):
        population_size = 16 * self.dim # Slightly larger population for diversity
        F = 0.7 # Adjusted for more aggressive mutations
        CR = 0.85 # Slightly reduced CR for exploration
        delta_F = 0.08 # Increased delta_F for dynamic adaptation
        delta_CR = 0.04 # Slightly increased delta_CR
        epsilon = 0.002 # Adjusted for infrequent innovative jumps

        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        while evaluations < self.num_de:
            for i in range(population_size):
                if np.random.rand() < epsilon:
                    idxs = np.random.choice(list(set(range(population_size)) - {i}), 4, replace=False)
                    weights = np.random.dirichlet([1.2]*4)
                    mutant = np.clip(np.dot(weights, population[idxs]), self.lower_bound, self.upper_bound)
                else:
                    idxs = np.random.choice(list(set(range(population_size)) - {i}), 3, replace=False)
                    a, b, c = population[idxs]
                    mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial
                    F = min(1.0, F + delta_F * 0.35)
                    CR = min(1.0, CR + delta_CR * 0.45)
                else:
                    F = max(0.1, F - delta_F * 0.5)
                    CR = max(0.15, CR - delta_CR * 0.6)

                if evaluations >= self.num_de:
                    break

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]

        if evaluations < self.budget:
            result = minimize(func, best_solution, method='Nelder-Mead',
                              bounds=[(self.lower_bound, self.upper_bound)] * self.dim,
                              options={'maxiter': self.num_nm, 'adaptive': True, 'xatol': 1e-6, 'fatol': 1e-6})
            evaluations += result.nfev
            if result.fun < fitness[best_idx]:
                best_solution = result.x

        return best_solution
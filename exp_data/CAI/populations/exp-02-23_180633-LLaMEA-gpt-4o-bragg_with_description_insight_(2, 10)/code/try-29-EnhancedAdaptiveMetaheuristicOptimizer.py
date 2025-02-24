import numpy as np
from scipy.optimize import minimize

class EnhancedAdaptiveMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.bounds = None

    def adaptive_quasi_oppositional_initialization(self, lb, ub):
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        quasi_opposite_population = lb + ub - population
        combined_population = np.concatenate((population, quasi_opposite_population), axis=0)
        return combined_population

    def dynamic_periodicity_enforcement(self, solution):
        period_length = self.dim // 2
        for i in range(period_length):
            solution[i + period_length] = solution[i]
        return solution

    def adaptive_differential_evolution(self, func):
        np.random.seed(42)
        population = self.adaptive_quasi_oppositional_initialization(self.bounds.lb, self.bounds.ub)
        population_fitness = np.array([func(ind) for ind in population])

        F_base, CR_base = 0.8, 0.7
        F_decay, CR_growth = 0.99, 1.01

        for _ in range(self.budget // (2 * self.population_size)):
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                F = F_base * (F_decay ** _)
                CR = min(1.0, CR_base * (CR_growth ** _))
                mutant = np.clip(a + F * (b - c), self.bounds.lb, self.bounds.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, population[i])
                trial = self.dynamic_periodicity_enforcement(trial)

                trial_fitness = func(trial)
                if trial_fitness < population_fitness[i]:
                    population[i] = trial
                    population_fitness[i] = trial_fitness

            best_idx = np.argmin(population_fitness)
            best_individual = population[best_idx]

        return best_individual

    def optimized_local_refinement(self, func, initial_guess):
        iter_count = 0
        res = minimize(func, initial_guess, method='L-BFGS-B', bounds=list(zip(self.bounds.lb, self.bounds.ub)))
        while not res.success and iter_count < 3:  # Retry mechanism for local refinement
            iter_count += 1
            res = minimize(func, res.x, method='L-BFGS-B', bounds=list(zip(self.bounds.lb, self.bounds.ub)))
        return res.x if res.success else initial_guess

    def __call__(self, func):
        self.bounds = func.bounds
        best_global_solution = self.adaptive_differential_evolution(func)
        best_solution = self.optimized_local_refinement(func, best_global_solution)
        return best_solution
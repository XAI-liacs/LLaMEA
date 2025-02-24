import numpy as np
from scipy.optimize import minimize

class EnhancedHybridMetaheuristicOptimizer:
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

    def periodicity_constraint(self, solution, generation, max_generations):
        period_length = self.dim // 2
        blend_ratio = 0.6 + 0.3 * (generation / max_generations)  # Dynamically adjusted blending
        for i in range(period_length):
            solution[i + period_length] = blend_ratio * solution[i] + (1 - blend_ratio) * solution[i + period_length]
        return solution

    def differential_evolution(self, func):
        np.random.seed(42)
        population = self.adaptive_quasi_oppositional_initialization(self.bounds.lb, self.bounds.ub)
        population_fitness = np.array([func(ind) for ind in population])

        for generation in range(self.budget // (2 * self.population_size)):
            if generation % 10 == 0:  # Dynamic population adjustment
                self.population_size = max(int(self.population_size * 0.9), 10)
            F = 0.5 + (0.9 - 0.5) * generation / (self.budget // (2 * self.population_size))
            CR = 0.5 + (0.9 - 0.5) * generation / (self.budget // (2 * self.population_size))
            best_idx = np.argmin(population_fitness)
            best_individual = population[best_idx]
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                # Introduce targeted mutation
                target = best_individual if np.random.rand() < 0.3 else a
                mutant = np.clip(target + F * (b - c), self.bounds.lb, self.bounds.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, population[i])
                trial = self.periodicity_constraint(trial, generation, self.budget // (2 * self.population_size))

                trial_fitness = func(trial)
                if trial_fitness < population_fitness[i]:
                    population[i] = trial
                    population_fitness[i] = trial_fitness

            if np.random.rand() < 0.6:  # Probabilistic local optimization
                best_idx = np.argmin(population_fitness)
                best_individual = population[best_idx]
                best_individual = self.local_optimization(func, best_individual)

        return best_individual

    def local_optimization(self, func, initial_guess):
        res = minimize(func, initial_guess, method='L-BFGS-B', bounds=list(zip(self.bounds.lb, self.bounds.ub)))
        return res.x if res.success else initial_guess

    def __call__(self, func):
        self.bounds = func.bounds
        best_global_solution = self.differential_evolution(func)
        best_solution = self.local_optimization(func, best_global_solution)
        return best_solution
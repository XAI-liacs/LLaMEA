import numpy as np
from scipy.optimize import minimize

class SymbioticOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4, 10 * dim)
        self.bounds = None
        self.adaptive_factor = 0.7  # Increased from 0.5 to 0.7
        self.penalty_decay = 0.95  # New decay factor for periodicity penalty

    def initialize_population(self):
        lb, ub = self.bounds.lb, self.bounds.ub
        return lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def symbiotic_interaction(self, population):
        new_population = np.copy(population)
        for i in range(0, self.population_size, 2):
            if i + 1 < self.population_size:
                ind1, ind2 = population[i], population[i + 1]
                new_population[i] = self.adaptive_crossover(ind1, ind2)
                new_population[i + 1] = self.adaptive_crossover(ind2, ind1)
        return new_population

    def adaptive_crossover(self, ind1, ind2):
        diff = ind2 - ind1
        new_ind = ind1 + self.adaptive_factor * diff
        return np.clip(new_ind, self.bounds.lb, self.bounds.ub)

    def periodicity_penalty(self, solution):
        period = self.dim // 2
        penalties = [(solution[i] - solution[i + period]) ** 2 for i in range(self.dim - period)]
        return np.sum(penalties) * self.penalty_decay  # Apply decay

    def __call__(self, func):
        self.bounds = func.bounds
        population = self.initialize_population()
        best_solution = None
        best_score = float('inf')

        eval_count = 0
        while eval_count < self.budget:
            population = self.symbiotic_interaction(population)
            new_population = np.zeros_like(population)

            for i in range(self.population_size):
                candidate = population[i]
                candidate_score = func(candidate) + self.periodicity_penalty(candidate)
                eval_count += 1

                if candidate_score < best_score:
                    best_score = candidate_score
                    best_solution = candidate

                new_population[i] = candidate

            population = new_population

            # Local optimization using BFGS if budget allows
            if eval_count < self.budget:
                result = minimize(func, best_solution, method='L-BFGS-B', bounds=zip(self.bounds.lb, self.bounds.ub))
                eval_count += result.nfev
                if result.fun < best_score:
                    best_score = result.fun
                    best_solution = result.x

        return best_solution
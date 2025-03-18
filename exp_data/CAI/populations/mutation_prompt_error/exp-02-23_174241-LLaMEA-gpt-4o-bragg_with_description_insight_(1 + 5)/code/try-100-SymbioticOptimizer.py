import numpy as np
from scipy.optimize import minimize

class SymbioticOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4, 10 * dim)
        self.bounds = None
        self.adaptive_factor = 0.8  # Slightly increase adaptive factor for broader exploration
        self.mutation_factor = 0.5  # Added mutation factor for differential evolution
        self.crossover_prob = 0.7   # Crossover probability for differential evolution

    def initialize_population(self):
        lb, ub = self.bounds.lb, self.bounds.ub
        return lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def symbiotic_interaction(self, population):
        new_population = np.copy(population)
        for i in range(0, self.population_size, 2):
            if i + 1 < self.population_size:
                ind1, ind2 = population[i], population[i + 1]
                new_population[i] = self.adaptive_crossover(ind1, ind2)
                new_population[i + 1] = self.differential_mutation(ind1, ind2, population)
        return new_population

    def adaptive_crossover(self, ind1, ind2):
        diff = ind2 - ind1
        new_ind = ind1 + self.adaptive_factor * diff
        return np.clip(new_ind, self.bounds.lb, self.bounds.ub)

    def differential_mutation(self, ind1, ind2, population):
        idxs = np.random.choice(np.arange(self.population_size), 3, replace=False)
        a, b, c = population[idxs]
        mutant = a + self.mutation_factor * (b - c)
        cross_points = np.random.rand(self.dim) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, ind1)
        return np.clip(trial, self.bounds.lb, self.bounds.ub)

    def periodicity_penalty(self, solution):
        period = self.dim // 2
        penalties = [(solution[i] - solution[i + period]) ** 2 for i in range(self.dim - period)]
        return np.sum(penalties) * 0.1  # Reduced penalty weight for flexibility

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
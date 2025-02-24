import numpy as np
from scipy.optimize import minimize

class HybridBraggOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_probability = 0.7
        self.local_refinement_budget = int(budget * 0.1)  # 10% of the budget for local refinement
        self.global_budget = budget - self.local_refinement_budget  # Remaining budget for global search

    def differential_evolution(self, func, bounds):
        # Initialize population with periodic solutions
        population = self.initialize_population(bounds)
        scores = np.array([func(individual) for individual in population])
        for _ in range(self.global_budget // self.population_size):
            for i in range(self.population_size):
                # Mutation and crossover
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), bounds.lb, bounds.ub)
                crossover = np.random.rand(self.dim) < self.crossover_probability
                trial = np.where(crossover, mutant, population[i])
                # Selection
                trial_score = func(trial)
                if trial_score < scores[i]:
                    population[i], scores[i] = trial, trial_score
        return population[np.argmin(scores)], np.min(scores)

    def initialize_population(self, bounds):
        # Periodic initialization with small randomness
        periodic_solution = np.tile([(bounds.ub - bounds.lb) / 2 + bounds.lb], self.dim // 2)
        periodic_solution = np.concatenate((periodic_solution, periodic_solution[:self.dim % 2]))
        population = np.array([periodic_solution + 0.1 * np.random.randn(self.dim) for _ in range(self.population_size)])
        return np.clip(population, bounds.lb, bounds.ub)

    def local_refinement(self, func, x0, bounds):
        options = {'maxiter': self.local_refinement_budget, 'disp': False}
        res = minimize(func, x0, method='L-BFGS-B', bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)], options=options)
        return res.x, res.fun

    def __call__(self, func):
        bounds = func.bounds
        best_global_solution, best_score = self.differential_evolution(func, bounds)
        
        # Apply local search around the best found solution
        refined_solution, refined_score = self.local_refinement(func, best_global_solution, bounds)
        if refined_score < best_score:
            best_global_solution, best_score = refined_solution, refined_score
        
        return best_global_solution, best_score
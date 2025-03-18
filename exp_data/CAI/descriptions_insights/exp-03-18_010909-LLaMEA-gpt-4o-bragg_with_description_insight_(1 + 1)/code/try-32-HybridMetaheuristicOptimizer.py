import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20  # Population size for Differential Evolution
        self.F = 0.8  # Mutation factor for DE
        self.CR = 0.9  # Crossover probability for DE
        self.local_search_probability = 0.5  # Probability to invoke local search
        self.penalty_factor = 100.0  # Penalty factor for non-periodicity

    def differential_evolution(self, func, bounds):
        # Initialize population
        population = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evals = self.pop_size

        while num_evals < self.budget:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                # Mutation and crossover
                a, b, c = population[np.random.choice(self.pop_size, 3, replace=False)]
                adaptive_F = self.F * (1 - num_evals / self.budget) ** 0.5  # Enhanced adaptive mutation factor
                mutant = np.clip(a + adaptive_F * (b - c), bounds.lb, bounds.ub)
                cross_points = np.random.rand(self.dim) < (self.CR * (1 - num_evals / self.budget) ** 0.5)  # Further adaptive crossover rate adjustment
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Apply periodicity penalty
                periodicity_penalty = self.calculate_periodicity_penalty(trial)
                trial_fitness = func(trial) + periodicity_penalty * (1 - fitness[i] / np.min(fitness))  # Adaptive penalty scaling
                num_evals += 1

                # Selection
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness

                # Local search
                if np.random.rand() < self.local_search_probability * (1 - num_evals / self.budget):  # Dynamic local search probability
                    local_res = minimize(func, trial, method='L-BFGS-B', bounds=np.array(list(zip(bounds.lb, bounds.ub))))
                    local_fitness = local_res.fun
                    num_evals += local_res.nfev
                    if local_fitness < fitness[i]:
                        new_population[i] = local_res.x
                        fitness[i] = local_fitness

                if num_evals >= self.budget:
                    break
            population = new_population
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def calculate_periodicity_penalty(self, solution):
        # Define periodic structures and compare
        period = 2
        penalty = 0
        for i in range(0, len(solution) - period):
            penalty += np.sum((solution[i:i+period] - solution[i+period:i+2*period])**2)
        return self.penalty_factor * penalty

    def __call__(self, func):
        bounds = func.bounds
        best_solution, best_fitness = self.differential_evolution(func, bounds)
        return best_solution, best_fitness
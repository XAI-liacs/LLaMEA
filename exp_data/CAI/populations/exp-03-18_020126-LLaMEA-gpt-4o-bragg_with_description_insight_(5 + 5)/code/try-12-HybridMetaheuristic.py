import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.bounds = None

    def initialize_population(self, lb, ub):
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        opp_pop = lb + ub - pop  # Quasi-Oppositional initialization
        return np.vstack((pop, opp_pop))

    def evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def select_best(self, population, fitness):
        idx = np.argmin(fitness)
        return population[idx], fitness[idx]

    def enforce_periodicity(self, candidate):
        # Modified periodicity for every third layer instead of half-dimension
        for i in range(0, self.dim, 3):
            candidate[i:i+3] = np.mean(candidate[i:i+3])
        return candidate

    def crossover_and_mutate(self, target, mutant, cr=0.9):
        cross_points = np.random.rand(self.dim) < cr
        offspring = np.where(cross_points, mutant, target)
        return self.enforce_periodicity(offspring)

    def differential_evolution(self, func):
        lb, ub = self.bounds.lb, self.bounds.ub
        population = self.initialize_population(lb, ub)
        fitness = self.evaluate_population(population, func)
        eval_count = len(population)

        while eval_count < self.budget:
            for i in range(len(population)):
                if eval_count >= self.budget:
                    break
                indices = [idx for idx in range(len(population)) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                # Adaptive mutation factor based on evaluation count
                F = 0.5 + (0.5 * np.sin(np.pi * eval_count / self.budget))
                mutant = np.clip(a + F * (b - c), lb, ub)
                trial = self.crossover_and_mutate(population[i], mutant)
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness

        return self.select_best(population, fitness)

    def local_optimization(self, best_solution, func):
        result = minimize(func, best_solution, method='L-BFGS-B', bounds=list(zip(self.bounds.lb, self.bounds.ub)))
        return result.x if result.success else best_solution

    def __call__(self, func):
        self.bounds = func.bounds
        best_solution, _ = self.differential_evolution(func)
        best_solution = self.local_optimization(best_solution, func)
        return best_solution
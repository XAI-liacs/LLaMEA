import numpy as np
from scipy.optimize import minimize

class HybridBraggOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.current_evaluations = 0
        self.bounds = None

    def initialize_population(self):
        lb, ub = self.bounds.lb, self.bounds.ub
        population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        return population

    def evaluate_population(self, population, func):
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += len(population)
        return fitness

    def differential_evolution_step(self, population, fitness):
        F = 0.5
        CR = 0.9
        new_population = np.zeros_like(population)
        for i in range(self.population_size):
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), self.bounds.lb, self.bounds.ub)
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            new_population[i] = trial
        new_fitness = self.evaluate_population(new_population, func)
        for i in range(self.population_size):
            if new_fitness[i] < fitness[i]:
                population[i] = new_population[i]
                fitness[i] = new_fitness[i]
        return population, fitness

    def local_optimization(self, best_solution, func):
        result = minimize(func, best_solution, method='BFGS', bounds=list(zip(self.bounds.lb, self.bounds.ub)))
        return result.x, result.fun

    def encourage_periodicity(self, population):
        for i in range(self.population_size):
            population[i] = np.tile(population[i][:self.dim//2], 2)
        return population

    def __call__(self, func):
        self.bounds = func.bounds
        population = self.initialize_population()
        fitness = self.evaluate_population(population, func)
        while self.current_evaluations < self.budget:
            population = self.encourage_periodicity(population)
            population, fitness = self.differential_evolution_step(population, fitness)
            if self.current_evaluations + self.population_size >= self.budget:
                break

        best_idx = np.argmin(fitness)
        best_solution, best_fitness = population[best_idx], fitness[best_idx]
        if self.current_evaluations + 10 <= self.budget:
            best_solution, best_fitness = self.local_optimization(best_solution, func)
        
        return best_solution, best_fitness
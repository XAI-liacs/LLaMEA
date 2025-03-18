import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = None
        self.func_evals = 0
        self.archive = []  # Archive to store elite solutions

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population_size = min(self.initial_population_size, self.budget // 2)  # Adaptive population sizing
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            self.population[i] = self.periodic_constraint(self.population[i])

    def differential_evolution_step(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        new_population = np.copy(self.population)
        adaptive_mutation_factor = self.mutation_factor * (1 - self.func_evals / self.budget)
        diversity_factor = np.std(self.population, axis=0).mean()  # Calculate population diversity
        dynamic_crossover_rate = self.crossover_rate * (0.5 + diversity_factor)  # Adapt crossover rate
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = self.population[indices]
            mutant = np.clip(a + adaptive_mutation_factor * (b - c), lb, ub)
            crossover_mask = np.random.rand(self.dim) < dynamic_crossover_rate
            trial = np.where(crossover_mask, mutant, self.population[i])
            trial = np.clip(trial, lb, ub)
            if self.evaluate(trial) < self.evaluate(self.population[i]):
                new_population[i] = trial
        self.population = new_population
        self.dynamic_resize_population()  # Adjust the population size dynamically

    def evaluate(self, individual):
        if self.func_evals >= self.budget:
            return float('inf')
        self.func_evals += 1
        value = self.func(individual)
        self.archive.append((value, individual))  # Maintain an archive of evaluated solutions
        return value

    def local_optimization(self, x0, bounds):
        lb, ub = bounds.lb, bounds.ub
        result = minimize(self.func, x0, bounds=np.array([0.9 * lb + 0.1 * x0, 0.9 * ub + 0.1 * x0]).T, method='L-BFGS-B')
        if not result.success:
            result = minimize(self.func, x0, bounds=np.array([0.95 * lb + 0.05 * x0, 0.95 * ub + 0.05 * x0]).T, method='TNC')  # Enhanced local search
        return result.x if result.success else x0

    def periodic_constraint(self, individual):
        period = self.dim // 4
        for i in range(period, self.dim):
            individual[i] = individual[i % period]
        return individual

    def dynamic_resize_population(self):  # New function to dynamically resize population
        if self.func_evals < self.budget / 2:
            self.population_size = int(self.initial_population_size * (1 + 0.1 * np.sin(2 * np.pi * self.func_evals / self.budget)))
        elif self.func_evals > 3 * self.budget / 4:
            self.population_size = max(5, int(self.population_size * 0.9))

    def __call__(self, func):
        self.func = func
        bounds = func.bounds
        self.initialize_population(bounds)
        best_solution = None
        best_value = float('inf')

        while self.func_evals < self.budget:
            self.differential_evolution_step(bounds)
            for i in range(self.population_size):
                constrained_individual = self.periodic_constraint(self.population[i])
                constrained_individual = self.local_optimization(constrained_individual, bounds)
                self.population[i] = constrained_individual
                current_value = self.evaluate(constrained_individual)
                if current_value < best_value:
                    best_value = current_value
                    best_solution = constrained_individual
        
        self.archive.sort()  # Sort archive based on function values
        elite_solution = self.archive[0][1]  # Extract the best from archive
        return elite_solution
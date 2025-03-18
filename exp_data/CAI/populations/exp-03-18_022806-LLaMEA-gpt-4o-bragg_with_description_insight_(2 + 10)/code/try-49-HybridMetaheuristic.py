import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = None
        self.func_evals = 0

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            self.population[i] = self.periodic_constraint(self.population[i])

    def differential_evolution_step(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        new_population = np.copy(self.population)
        dynamic_crossover_rate = self.crossover_rate - 0.5 * (self.func_evals / self.budget)
        dynamic_mutation_factor = self.mutation_factor + 0.4 * (self.func_evals / self.budget)
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = self.population[indices]
            mutant = np.clip(a + dynamic_mutation_factor * (b - c) + np.random.normal(0, 0.1, size=self.dim), lb, ub)  # Added diversity
            crossover_mask = np.random.rand(self.dim) < dynamic_crossover_rate
            trial = np.where(crossover_mask, mutant, self.population[i])
            trial = np.clip(trial, lb, ub)
            if self.evaluate(trial) < self.evaluate(self.population[i]):
                new_population[i] = trial
        self.population = new_population

    def evaluate(self, individual):
        if self.func_evals >= self.budget:
            return float('inf')
        self.func_evals += 1
        return self.func(individual)

    def local_optimization(self, x0, bounds):
        lb, ub = bounds.lb, bounds.ub
        result = minimize(self.func, x0, bounds=np.array([0.9 * lb + 0.1 * x0, 0.9 * ub + 0.1 * x0]).T, method='L-BFGS-B')
        return result.x if result.success else x0

    def periodic_constraint(self, individual):
        period = self.dim // 2
        for i in range(period, self.dim):
            individual[i] = individual[i % period]
        return individual

    def adaptive_population_resize(self):
        factor = 1 + (0.5 * (self.func_evals / self.budget))  # Adaptive resizing
        self.population_size = int(self.population_size * factor)
        self.population_size = max(4, min(self.population_size, 20 * self.dim))  # Maintain reasonable limits

    def __call__(self, func):
        self.func = func
        bounds = func.bounds
        self.initialize_population(bounds)
        best_solution = None
        best_value = float('inf')

        while self.func_evals < self.budget:
            self.differential_evolution_step(bounds)
            self.adaptive_population_resize()  # Call to resize population
            for i in range(self.population_size):
                constrained_individual = self.periodic_constraint(self.population[i])
                constrained_individual = self.local_optimization(constrained_individual, bounds)
                self.population[i] = constrained_individual
                current_value = self.evaluate(constrained_individual)
                if current_value < best_value:
                    best_value = current_value
                    best_solution = constrained_individual
        
        return best_solution
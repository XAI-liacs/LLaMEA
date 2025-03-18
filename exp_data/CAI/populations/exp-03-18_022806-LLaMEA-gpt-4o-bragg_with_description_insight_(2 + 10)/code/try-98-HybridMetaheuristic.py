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
        diversity_factor = np.std(self.population, axis=0).mean()
        dynamic_crossover_rate = self.crossover_rate * (0.5 + diversity_factor)
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

    def evaluate(self, individual):
        if self.func_evals >= self.budget:
            return float('inf')
        self.func_evals += 1
        return self.func(individual)

    def local_optimization(self, x0, bounds):
        lb, ub = bounds.lb, bounds.ub
        result = minimize(self.func, x0, bounds=np.array([0.9 * lb + 0.1 * x0, 0.9 * ub + 0.1 * x0]).T, method='L-BFGS-B')
        if not result.success:
            result = minimize(self.func, x0, bounds=np.array([0.95 * lb + 0.05 * x0, 0.95 * ub + 0.05 * x0]).T, method='TNC')
        return result.x if result.success else x0

    def periodic_constraint(self, individual):
        period = self.dim // 4
        for i in range(period, self.dim):
            individual[i] = individual[i % period]
        return individual

    def __call__(self, func):
        self.func = func
        bounds = func.bounds
        self.initialize_population(bounds)
        best_solution = None
        best_value = float('inf')

        while self.func_evals < self.budget:
            self.differential_evolution_step(bounds)
            population_diversity = np.std(self.population, axis=0).mean()  # Added line
            dynamic_local_search_trigger = population_diversity > 0.05  # Added line
            for i in range(self.population_size):
                constrained_individual = self.periodic_constraint(self.population[i])
                if dynamic_local_search_trigger:  # Added line
                    constrained_individual = self.local_optimization(constrained_individual, bounds)
                self.population[i] = constrained_individual
                current_value = self.evaluate(constrained_individual)
                if current_value < best_value:
                    best_value = current_value
                    best_solution = constrained_individual
        
        return best_solution
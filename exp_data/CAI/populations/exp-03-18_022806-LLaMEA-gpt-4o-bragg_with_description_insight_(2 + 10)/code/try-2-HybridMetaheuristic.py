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

    def differential_evolution_step(self, bounds):
        lb, ub = bounds.lb, ub
        new_population = np.copy(self.population)
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = self.population[indices]
            mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
            crossover_mask = np.random.rand(self.dim) < self.crossover_rate
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
        result = minimize(self.func, x0, bounds=np.array([lb, ub]).T, method='L-BFGS-B')
        return result.x if result.success else x0

    def periodic_constraint(self, individual):
        # Improved periodicity enforcement
        period = self.dim // 2
        for i in range(period, self.dim):
            individual[i] = 0.5 * (individual[i] + individual[i % period])
        return individual

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
        
        return best_solution
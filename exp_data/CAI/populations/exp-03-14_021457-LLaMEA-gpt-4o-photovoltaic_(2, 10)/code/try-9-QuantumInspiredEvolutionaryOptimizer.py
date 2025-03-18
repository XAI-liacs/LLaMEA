import numpy as np

class QuantumInspiredEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.best_solution = None
        self.best_value = float('inf')
        
    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def evaluate_population(self, func, population):
        values = np.array([func(ind) for ind in population])
        return values
    
    def superposition(self, population):
        mean = np.mean(population, axis=0)
        std_dev = np.std(population, axis=0)
        return np.random.normal(mean, std_dev, (self.population_size, self.dim))
    
    def update_best(self, population, values):
        min_idx = np.argmin(values)
        if values[min_idx] < self.best_value:
            self.best_value = values[min_idx]
            self.best_solution = population[min_idx]
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        evaluations = 0
        
        while evaluations < self.budget:
            values = self.evaluate_population(func, population)
            self.update_best(population, values)
            if evaluations + self.population_size > self.budget:
                break
            population = self.superposition(population)
            population = np.clip(population, lb, ub)
            evaluations += self.population_size
            self.population_size = max(5, self.population_size - 1)  # Dynamically adjust population size
        
        return self.best_solution, self.best_value
import numpy as np

class QuantumInspiredOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 10
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += population_size

        while self.evaluations < self.budget:
            # Adaptive neighborhood size
            neighborhood_size = (ub - lb) * (0.1 + 0.8 * (1 - self.evaluations / self.budget))
            
            # Quantum-inspired superposition and collapse
            for i in range(population_size):
                if self.evaluations >= self.budget:
                    break
                # Create a new candidate solution by quantum jump
                new_candidate = population[i] + np.random.uniform(-1, 1, self.dim) * neighborhood_size
                new_candidate = np.clip(new_candidate, lb, ub)
                new_fitness = func(new_candidate)
                self.evaluations += 1
                
                # Selection: Replace if the new candidate is better
                if new_fitness < fitness[i]:
                    population[i] = new_candidate
                    fitness[i] = new_fitness

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]
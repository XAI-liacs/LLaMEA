import numpy as np
import random
import math

class HybridGA_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        def simulated_annealing(current, T, func):
            while T > 1e-3:
                candidate = current + np.random.normal(0, T, self.dim)
                candidate = np.clip(candidate, -5.0, 5.0)
                delta_E = func(candidate) - func(current)
                if delta_E < 0 or np.random.rand() < math.exp(-delta_E / T):
                    current = candidate
                T *= 0.995
            return current
        
        population_size = 50
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(population_size)]
        
        for _ in range(self.budget):
            offspring = []
            for ind in population:
                mutation = ind + np.random.normal(0, 0.1, self.dim)
                mutation = np.clip(mutation, -5.0, 5.0)
                
                if func(mutation) < func(ind):
                    offspring.append(mutation)
                else:
                    offspring.append(ind)
            
            offspring.sort(key=lambda x: func(x))
            population[:population_size//2] = offspring[:population_size//2]
            
            for i in range(population_size//2, population_size):
                population[i] = simulated_annealing(population[i], 1.0, func)
        
        population.sort(key=lambda x: func(x))
        return population[0]
import numpy as np

class EnhancedChaoticOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def initialize_population(self, lb, ub, size, chaotic_map):
        population = np.zeros((size, self.dim))
        for i in range(size):
            r = np.random.rand(self.dim)
            if chaotic_map == "logistic":
                population[i] = lb + (ub - lb) * (3.9 * r * (1 - r))
            elif chaotic_map == "sine":
                population[i] = lb + (ub - lb) * (np.sin(np.pi * r))
        return np.clip(population, lb, ub)
    
    def chaotic_hybrid_search(self, func, lb, ub):
        size = 5 + self.dim // 2
        population = self.initialize_population(lb, ub, size, chaotic_map="logistic")
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]
        
        evaluations = size
        
        while evaluations < self.budget:
            for i in range(size):
                perturbation_intensity = 0.2 * (1 - (evaluations / self.budget)**(1/4))  # Adjusted intensity exponent
                neighbor_radius = perturbation_intensity * (ub - lb)  # Dynamic neighborhood
                candidate = population[i] + np.random.uniform(-neighbor_radius, neighbor_radius, self.dim)
                candidate = np.clip(candidate, lb, ub)
                candidate_fitness = func(candidate)
                evaluations += 1
                acceptance_probability = 0.15 + (fitness[i] - candidate_fitness) / np.abs(fitness[i] + candidate_fitness + 1e-10)
                if candidate_fitness < fitness[i] or np.random.rand() < acceptance_probability:
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    if candidate_fitness < best_fitness:
                        best_fitness = candidate_fitness
                        best = candidate
                if evaluations >= self.budget:
                    break
            
            if evaluations % (self.budget // 5) == 0:
                scale = 1.0 - (evaluations / self.budget)
                new_population = self.initialize_population(lb, ub, size, chaotic_map="sine")
                new_population = best + scale * (new_population - best)
                new_population_fitness = np.array([func(ind) for ind in new_population])
                evaluations += size
                population = np.where(new_population_fitness < fitness, new_population, population)
                fitness = np.minimum(new_population_fitness, fitness)
                best_idx = np.argmin(fitness)
                if fitness[best_idx] < best_fitness:
                    best_fitness = fitness[best_idx]
                    best = population[best_idx]

        return best
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        return self.chaotic_hybrid_search(func, lb, ub)
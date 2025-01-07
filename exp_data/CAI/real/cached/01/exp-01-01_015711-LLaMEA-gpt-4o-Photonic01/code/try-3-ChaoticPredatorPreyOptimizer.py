import numpy as np

class ChaoticPredatorPreyOptimizer:
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

    def predator_prey_dynamics(self, population, fitness, best, lb, ub, func):
        size = len(population)
        for _ in range(size // 2):
            predator_idx = np.random.randint(size)
            prey_idx = np.random.randint(size)
            if fitness[predator_idx] > fitness[prey_idx]:
                continue
            candidate = population[predator_idx] + np.random.uniform(-0.2, 0.2, self.dim) * (best - population[prey_idx])
            candidate = np.clip(candidate, lb, ub)
            candidate_fitness = func(candidate)
            if candidate_fitness < fitness[prey_idx]:
                population[prey_idx] = candidate
                fitness[prey_idx] = candidate_fitness
        return population, fitness

    def chaotic_hybrid_search(self, func, lb, ub):
        size = 5 + self.dim // 2
        population = self.initialize_population(lb, ub, size, chaotic_map="logistic")
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]
        
        evaluations = size
        
        while evaluations < self.budget:
            population, fitness = self.predator_prey_dynamics(population, fitness, best, lb, ub, func)
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_fitness = fitness[best_idx]
                best = population[best_idx]

            if evaluations >= self.budget:
                break

            if evaluations % (self.budget // 10) == 0:
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
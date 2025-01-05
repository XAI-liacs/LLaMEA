import numpy as np

class SynergisticChaoticOptimizer:
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
                f = 0.8 + 0.2 * np.random.rand()  # Differential weight factor
                r1, r2 = np.random.randint(size, size=2)
                diff = f * (population[r1] - population[r2])
                candidate = population[i] + diff + np.random.normal(0, 0.05, self.dim)
                candidate = np.clip(candidate, lb, ub)
                candidate_fitness = func(candidate)
                evaluations += 1
                if candidate_fitness < fitness[i]:
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    if candidate_fitness < best_fitness:
                        best_fitness = candidate_fitness
                        best = candidate
                if evaluations >= self.budget:
                    break

            if evaluations % (self.budget // 10) == 0:
                new_population = self.initialize_population(lb, ub, size, chaotic_map="sine")
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
import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 15 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.bounds = None

    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                    
                # Select indices for mutation
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                # Mutation and crossover
                mutant = np.clip(population[a] + self.mutation_factor * (population[b] - population[c]), self.bounds[0], self.bounds[1])
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])
                
                # Evaluate trial vector
                trial_fitness = func(trial)
                evaluations += 1
                
                # Selection process
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update global best
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness
            
            # Adaptive mutation factor
            self.mutation_factor = 0.5 + 0.3 * np.sin(np.pi * evaluations / self.budget)
        
        return best_solution
import numpy as np

class DifferentialEvolutionaryStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.adaptive_pressure = 0.5
    
    def initialize_population(self, bounds):
        """ Initialize population within given bounds """
        return np.random.rand(self.population_size, self.dim) * (bounds.ub - bounds.lb) + bounds.lb
    
    def mutate(self, population, idx):
        """ Perform differential mutation """
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = population[a] + self.mutation_factor * (population[b] - population[c])
        return np.clip(mutant, bounds.lb, bounds.ub)
    
    def crossover(self, target, mutant):
        """ Perform crossover between target and mutant """
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        offspring = np.where(crossover_mask, mutant, target)
        return offspring
    
    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        fitness = np.array([func(individual) for individual in population])
        
        evaluations = self.population_size
        while evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(population, i)
                offspring = self.crossover(population[i], mutant)
                offspring_fitness = func(offspring)
                evaluations += 1
                
                # Selection based on fitness
                if offspring_fitness > fitness[i]:
                    population[i] = offspring
                    fitness[i] = offspring_fitness
            
                if evaluations >= self.budget:
                    break
            
            # Adjust mutation factor and crossover rate adaptively
            successful_offsprings = np.mean(fitness > np.median(fitness))
            self.mutation_factor = np.clip(self.mutation_factor + self.adaptive_pressure * (0.5 - successful_offsprings), 0.5, 1.0)
            self.crossover_rate = np.clip(self.crossover_rate + self.adaptive_pressure * successful_offsprings, 0.5, 1.0)
            
        best_idx = np.argmax(fitness)
        return population[best_idx]
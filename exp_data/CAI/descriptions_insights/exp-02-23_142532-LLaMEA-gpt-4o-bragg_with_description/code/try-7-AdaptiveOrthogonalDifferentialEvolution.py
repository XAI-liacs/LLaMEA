import numpy as np

class AdaptiveOrthogonalDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.orthogonal_size = 3
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
    
    def orthogonal_array(self):
        """ Generates an orthogonal array for diversified sampling """
        base = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        orthogonal_expansion = np.hstack([np.tile(base, (self.orthogonal_size, 1)), np.repeat(base, self.orthogonal_size, axis=0)])
        return orthogonal_expansion[:self.orthogonal_size, :self.dim]
    
    def initialize_population(self, bounds):
        """ Initialize a population within given bounds """
        return np.random.rand(self.population_size, self.dim) * (bounds.ub - bounds.lb) + bounds.lb
    
    def differential_mutation(self, target_idx, population, bounds):
        """ Perform differential mutation """
        idxs = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = population[np.random.choice(idxs, 3, replace=False)]
        mutant = a + self.F * (b - c)
        return np.clip(mutant, bounds.lb, bounds.ub)
    
    def crossover(self, target, mutant):
        """ Perform crossover between target and mutant vectors """
        crossover_mask = np.random.rand(self.dim) < self.CR
        offspring = np.where(crossover_mask, mutant, target)
        return offspring
    
    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        best_solution = None
        best_fitness = float('-inf')
        
        evaluations = 0
        while evaluations < self.budget:
            fitness = np.array([func(ind) for ind in population])
            evaluations += len(fitness)
            
            # Update best solution found
            max_idx = np.argmax(fitness)
            if fitness[max_idx] > best_fitness:
                best_fitness = fitness[max_idx]
                best_solution = population[max_idx]
            
            new_population = []
            for i in range(self.population_size):
                mutant = self.differential_mutation(i, population, bounds)
                offspring = self.crossover(population[i], mutant)
                new_population.append(offspring)
            
            # Apply orthogonal array sampling for added diversity
            orthogonal_array = self.orthogonal_array()
            for array in orthogonal_array:
                candidate = np.dot(array, new_population[:self.orthogonal_size]) / 2.0
                candidate = np.clip(candidate, bounds.lb, bounds.ub)
                new_population.append(candidate)

            new_population = np.array(new_population)
            population = new_population[:self.population_size]
        
        return best_solution
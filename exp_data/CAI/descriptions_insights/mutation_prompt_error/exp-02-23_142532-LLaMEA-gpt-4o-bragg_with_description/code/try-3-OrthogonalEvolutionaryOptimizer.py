import numpy as np

class OrthogonalEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.orthogonal_size = 4  # Fixed orthogonal size
        self.mutation_rate = 0.1
        self.cross_rate = 0.5
    
    def orthogonal_array(self):
        """ Generates an orthogonal array for diversified sampling """
        base = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        orthogonal_expansion = np.hstack([np.tile(base, (self.orthogonal_size, 1)), np.repeat(base, self.orthogonal_size, axis=0)])
        return orthogonal_expansion[:self.dim, :self.dim]
    
    def initialize_population(self, bounds):
        """ Initialize a population within given bounds """
        return np.random.rand(self.population_size, self.dim) * (bounds.ub - bounds.lb) + bounds.lb
    
    def crossover(self, parent1, parent2):
        """ Perform crossover between two parents """
        mask = np.random.rand(self.dim) < self.cross_rate
        offspring = np.where(mask, parent1, parent2)
        return offspring

    def mutate(self, individual, bounds):
        """ Mutate an individual with a given mutation rate """
        mutation_vector = np.random.randn(self.dim)
        mask = np.random.rand(self.dim) < self.mutation_rate
        individual[mask] += mutation_vector[mask]
        individual = np.clip(individual, bounds.lb, bounds.ub)
        return individual

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        orthogonal_array = self.orthogonal_array()
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
            
            # Select top individuals
            top_indices = np.argsort(fitness)[-self.orthogonal_size:]
            top_individuals = population[top_indices]

            # Generate new population using crossover and mutation
            new_population = []
            for parent1, parent2 in zip(top_individuals, np.roll(top_individuals, 1, axis=0)):
                offspring = self.crossover(parent1, parent2)
                offspring = self.mutate(offspring, bounds)
                new_population.append(offspring)

            # Include orthogonal array designs for diversity
            for array in orthogonal_array:
                candidate = np.dot(array, top_individuals) / 2.0
                candidate = np.clip(candidate, bounds.lb, bounds.ub)
                new_population.append(candidate)

            # Ensure new population doesn't exceed population size
            population = new_population[:self.population_size]
        
        return best_solution
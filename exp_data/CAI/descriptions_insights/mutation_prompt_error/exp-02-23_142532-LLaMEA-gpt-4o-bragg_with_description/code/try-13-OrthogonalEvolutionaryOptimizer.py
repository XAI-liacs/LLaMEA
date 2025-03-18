import numpy as np

class OrthogonalEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, dim)
        self.orthogonal_size = 3
        self.mutation_rate = 0.1
        self.cross_rate = 0.5

    def adaptive_parameters(self, evaluations):
        """ Adapt parameters based on evaluations to balance exploration and exploitation """
        scaling_factor = 1 - (evaluations / self.budget)
        self.mutation_rate = max(0.05, scaling_factor * 0.1)
        self.cross_rate = 0.5 + (0.5 * (1 - scaling_factor))

    def orthogonal_array(self):
        """ Generates an orthogonal array for diversified sampling """
        base = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        expansion_size = self.orthogonal_size * 2
        orthogonal_expansion = np.vstack([np.tile(base, (expansion_size, 1)), np.repeat(base, expansion_size, axis=0)])
        orthogonal_expansion = orthogonal_expansion[:self.dim, :self.dim]
        return orthogonal_expansion

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

            # Adapt parameters based on evaluations
            self.adaptive_parameters(evaluations)
            
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
            orthogonal_array = self.orthogonal_array()
            for array in orthogonal_array:
                candidate = np.dot(array, top_individuals) / self.orthogonal_size
                candidate = np.clip(candidate, bounds.lb, bounds.ub)
                new_population.append(candidate)

            # Ensure new population doesn't exceed population size
            population = new_population[:self.population_size]
        
        return best_solution
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a given budget and dimensionality.

        Parameters:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimize the black box function `func` using the given budget and search space.

        Parameters:
        func (function): The black box function to optimize.

        Returns:
        tuple: The optimized parameters and the optimized function value.
        """
        # Initialize the population size
        population_size = 100

        # Initialize the population with random parameters
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))

        # Evaluate the function for each individual in the population
        for _ in range(self.budget):
            # Evaluate the function for each individual in the population
            func_values = func(population)

            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(func_values)[::-1][:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Apply the mutation strategy
            for _ in range(self.population_size // 2):
                # Select a random individual
                individual = new_population[np.random.choice(population_size)]

                # Generate two new offspring by applying the mutation
                mutated_individual1 = individual + np.random.uniform(-1.0, 1.0, self.dim)
                mutated_individual2 = individual - np.random.uniform(-1.0, 1.0, self.dim)

                # Replace the old individual with the new offspring
                new_population[np.random.choice(population_size), :] = mutated_individual1
                new_population[np.random.choice(population_size), :] = mutated_individual2

            # Replace the old population with the new population
            population = new_population

        # Return the optimized parameters and the optimized function value
        return population, func(population)
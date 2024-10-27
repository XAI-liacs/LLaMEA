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

            # Apply crossover and inversion to refine the individual lines
            for i in range(population_size):
                parent1 = new_population[i]
                parent2 = new_population[(i + 1) % population_size]
                child = self.inversion(parent1, parent2)
                new_population[i] = child
                new_population[(i + 1) % population_size] = parent2

            # Replace the old population with the new population
            population = new_population

        # Return the optimized parameters and the optimized function value
        return population, func(population)

    def inversion(self, parent1, parent2):
        """
        Inversion mutation strategy that combines inversion and crossover.

        Parameters:
        parent1 (numpy array): The first parent individual.
        parent2 (numpy array): The second parent individual.

        Returns:
        numpy array: The child individual resulting from the inversion mutation.
        """
        # Swap the bits of the two parents
        child = np.concatenate((parent1, parent2[1:] + parent1[:-1]))
        return child
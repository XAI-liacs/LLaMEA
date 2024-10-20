import random
import numpy as np

class MetaDifferentialEvolution:
    def __init__(self, budget, dim):
        """
        Initialize the MetaDifferential Evolution algorithm.

        Parameters:
        - budget (int): The maximum number of function evaluations.
        - dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.01
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))
        self.refine = False

    def __call__(self, func):
        """
        Optimize the black box function using MetaDifferential Evolution.

        Parameters:
        - func (function): The black box function to optimize.

        Returns:
        - optimized_func (function): The optimized function.
        """
        while self.budget > 0:
            # Generate a new population by perturbing the current population
            new_population = self.generate_new_population()

            # Evaluate the new population using the given budget
            new_population_evaluations = np.random.randint(1, self.budget + 1)

            # Evaluate the new population
            new_population_evaluations = np.minimum(new_population_evaluations, self.budget)

            # Select the fittest individuals from the new population
            self.population = self.select_fittest(new_population, new_population_evaluations)

            # Update the population size
            self.population_size = min(self.population_size, len(new_population))

            # Check if the population has been fully optimized
            if len(self.population) == 0:
                break

            # Perform mutation on the fittest individuals
            self.population = self.mutate(self.population)

            # Refine the search strategy
            if not self.refine:
                self.refine = True
                # Select the fittest individuals using a weighted average of the current and refined search spaces
                fittest_individuals = np.minimum(self.population, self.refine_search_space(self.population, self.refine))
                self.population = self.select_fittest(fittest_individuals, np.minimum(fittest_individuals, self.budget))

            # Update the budget for the next iteration
            self.budget = min(self.budget * 0.9, 100)

        # Return the optimized function
        return func

    def generate_new_population(self):
        """
        Generate a new population by perturbing the current population.

        Returns:
        - new_population (numpy array): The new population.
        """
        new_population = self.population.copy()
        for _ in range(self.population_size // 2):
            # Perturb the current individual
            new_population[random.randint(0, self.dim - 1)] += random.uniform(-5.0, 5.0)

        return new_population

    def select_fittest(self, new_population, new_population_evaluations):
        """
        Select the fittest individuals from the new population.

        Parameters:
        - new_population (numpy array): The new population.
        - new_population_evaluations (numpy array): The evaluations of the new population.

        Returns:
        - fittest_population (numpy array): The fittest population.
        """
        # Calculate the fitness of each individual
        fitness = np.abs(new_population_evaluations)

        # Select the fittest individuals
        fittest_population = new_population[fitness.argsort()[:len(fitness)]]

        return fittest_population

    def refine_search_space(self, population, refine):
        """
        Refine the search space by adjusting the bounds and the mutation rate.

        Parameters:
        - population (numpy array): The population.
        - refine (bool): Whether to refine the search space.

        Returns:
        - refined_population (numpy array): The refined population.
        """
        # Refine the bounds
        bounds = np.minimum(population, 5.0)
        bounds = np.maximum(bounds, -5.0)
        population = bounds

        # Refine the mutation rate
        self.mutation_rate = np.random.uniform(0.01, 0.1)

        return population

# Description: Refines the MetaDifferential Evolution algorithm by incorporating a refinement step to adapt the search strategy.
# Code: 
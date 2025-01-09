import numpy as np

class GeneticSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.05
        self.temperature = 1.0
        self.min_temperature = 0.01
        self.cooling_rate = 0.9

    def _mutate(self, individual):
        mutated_individual = individual + np.random.uniform(-1, 1, self.dim) * self.mutation_rate
        mutated_individual = np.clip(mutated_individual, -5.0, 5.0)
        return mutated_individual

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        
        for _ in range(self.budget):
            # Perform tournament selection
            idx1, idx2 = np.random.choice(self.population_size, 2, replace=False)
            parent1, parent2 = population[idx1], population[idx2]

            # Crossover
            child = (parent1 + parent2) / 2.0

            # Mutation
            mutated_child = self._mutate(child)

            # Simulated Annealing
            old_fitness = func(child)
            new_fitness = func(mutated_child)
            acceptance_prob = np.exp((old_fitness - new_fitness) / self.temperature)

            if acceptance_prob > np.random.rand():
                population[idx1] = mutated_child

            # Cooling
            self.temperature = max(self.min_temperature, self.temperature * self.cooling_rate)

        # Return the best solution found
        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]
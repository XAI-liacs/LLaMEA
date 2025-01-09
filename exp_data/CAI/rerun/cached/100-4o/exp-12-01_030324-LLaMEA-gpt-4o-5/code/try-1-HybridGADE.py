import numpy as np

class HybridGADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.temperature = 1.0
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def differential_evolution_crossover(self, target, donor):
        trial = np.copy(target)
        for i in range(self.dim):
            if np.random.rand() < self.crossover_probability:
                trial[i] = donor[i]
        return trial

    def simulated_annealing_mutation(self, solution):
        mutated = np.copy(solution)
        for i in range(self.dim):
            if np.random.rand() < np.exp(-1.0 / self.temperature):
                mutated[i] += np.random.normal(0, 1)
                mutated[i] = np.clip(mutated[i], self.lower_bound, self.upper_bound)
        return mutated

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        # Main optimization loop
        while evaluations < self.budget:
            new_population = []
            for i in range(self.population_size):
                # Select individuals for crossover
                candidates = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[candidates]
                # Adaptive mutation factor based on remaining budget
                adaptive_mutation_factor = self.mutation_factor * (1 - evaluations / self.budget)
                donor = a + adaptive_mutation_factor * (b - c)
                donor = np.clip(donor, self.lower_bound, self.upper_bound)

                # Crossover
                trial = self.differential_evolution_crossover(population[i], donor)

                # Mutation
                trial = self.simulated_annealing_mutation(trial)

                # Evaluate trial solution
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                    fitness[i] = trial_fitness
                else:
                    new_population.append(population[i])

                if evaluations >= self.budget:
                    break

            population = np.array(new_population)

        # Return the best solution found
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]
import numpy as np

class HybridGADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.elitism_rate = 0.1
        self.bounds = (-5.0, 5.0)
        self.num_elites = int(self.elitism_rate * self.population_size)

    def __call__(self, func):
        population = np.random.uniform(
            self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            # Calculate dynamic mutation factor based on fitness diversity
            fitness_std = np.std(fitness)
            self.mutation_factor = 0.5 + 0.5 * (fitness_std / (np.mean(fitness) + 1e-9))

            # Select elites
            elite_indices = np.argsort(fitness)[:self.num_elites]
            elites = population[elite_indices]

            # Differential Evolution Mutation
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + self.mutation_factor * (x2 - x3), *self.bounds)

                # Crossover
                trial = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_rate:
                        trial[j] = mutant[j]

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Genetic Algorithm Crossover and Elitism
            new_population = []
            sorted_indices = np.argsort(fitness)
            for i in range(self.num_elites):
                new_population.append(population[sorted_indices[i]])

            while len(new_population) < self.population_size:
                if evaluations >= self.budget:
                    break
                parents_indices = np.random.choice(self.population_size, 2, replace=False)
                parents = population[parents_indices]
                crossover_point = np.random.randint(1, self.dim)
                offspring = np.concatenate(
                    (parents[0][:crossover_point], parents[1][crossover_point:]))

                if np.random.rand() < 0.5:  # Mutation in GA
                    mut_idx = np.random.randint(self.dim)
                    offspring[mut_idx] += np.random.normal(0, 0.1)
                    offspring = np.clip(offspring, *self.bounds)

                new_fitness = func(offspring)
                evaluations += 1
                new_population.append(offspring)

            population = np.array(new_population)
            fitness = np.array([func(ind) for ind in population])
            evaluations += self.population_size

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_rate = 0.5
        self.crossover_rate = 0.9
        self.mutation_rate_adapt = 0.1
        self.crossover_rate_adapt = 0.1

    def initialize_population(self, bounds):
        """ Initialize a population within given bounds """
        return np.random.rand(self.population_size, self.dim) * (bounds.ub - bounds.lb) + bounds.lb

    def differential_mutation(self, population, best_idx):
        """ Perform differential mutation on the population """
        idxs = [i for i in range(self.population_size) if i != best_idx]
        a, b, c = population[np.random.choice(idxs, 3, replace=False)]
        mutant = a + self.mutation_rate * (b - c)
        return np.clip(mutant, bounds.lb, bounds.ub)

    def crossover(self, target, mutant):
        """ Perform crossover between target and mutant vectors """
        trial = np.copy(target)
        for i in range(self.dim):
            if np.random.rand() < self.crossover_rate:
                trial[i] = mutant[i]
        return trial

    def adapt_parameters(self):
        """ Adapt mutation and crossover rates """
        self.mutation_rate = max(0.1, min(0.9, self.mutation_rate + np.random.randn() * self.mutation_rate_adapt))
        self.crossover_rate = max(0.1, min(0.9, self.crossover_rate + np.random.randn() * self.crossover_rate_adapt))

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
                if i == max_idx:
                    new_population.append(population[i])
                    continue
                mutant = self.differential_mutation(population, max_idx)
                trial = self.crossover(population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness > fitness[i]:
                    new_population.append(trial)
                else:
                    new_population.append(population[i])

            population = np.array(new_population)

            # Adapt parameters
            self.adapt_parameters()

        return best_solution
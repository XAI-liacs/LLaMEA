import numpy as np

class PeriodicDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub

        # Initialize population
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Differential evolution mutation
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)

                # Crossover
                trial = np.copy(population[i])
                crossover_points = np.random.rand(self.dim) < self.crossover_rate
                trial[crossover_points] = mutant[crossover_points]

                # Periodicity encouragement by shifting elements
                if np.random.rand() < 0.5:
                    shift = np.random.randint(1, self.dim // 2)
                    trial[:shift] = trial[-shift:]

                # Evaluate trial individual
                trial_fitness = func(trial)
                self.evaluations += 1

                # Selection
                if trial_fitness > fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

        best_index = np.argmax(fitness)
        return population[best_index]
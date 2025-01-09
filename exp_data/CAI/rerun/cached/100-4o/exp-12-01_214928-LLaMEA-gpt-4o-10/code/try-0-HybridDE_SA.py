import numpy as np

class HybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + int(1.5 * dim)  # typically a larger population for DE
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.temp = 100.0  # initial temperature for SA
        self.cooling_rate = 0.99
        self.evals = 0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evals += self.population_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        while self.evals < self.budget:
            for i in range(self.population_size):
                # Differential evolution mutation
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = population[np.random.choice(candidates, 3, replace=False)]
                mutant = a + self.mutation_factor * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                crossover = np.random.rand(self.dim) < self.crossover_probability
                trial = np.where(crossover, mutant, population[i])

                # Simulated annealing acceptance
                trial_fitness = func(trial)
                self.evals += 1
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temp):
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Update best solution
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

            # Cooling schedule
            self.temp *= self.cooling_rate

        return best_solution, best_fitness
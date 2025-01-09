import numpy as np

class HybridDEGaussian:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(10 * dim, budget // dim)
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.current_evals = 0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        self.current_evals += self.population_size

        while self.current_evals < self.budget:
            for i in range(self.population_size):
                if self.current_evals >= self.budget:
                    break

                # DE mutation
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # DE crossover
                crossover = np.random.rand(self.dim) < self.CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])

                # Gaussian random walk for exploitation
                if np.random.rand() < 0.2:  # 20% chance to apply Gaussian walk
                    step_size = (self.upper_bound - self.lower_bound) * 0.05
                    trial = trial + np.random.normal(0, step_size, self.dim)
                    trial = np.clip(trial, self.lower_bound, self.upper_bound)

                # Evaluate the trial solution
                trial_fitness = func(trial)
                self.current_evals += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

        # Return the best solution found
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]
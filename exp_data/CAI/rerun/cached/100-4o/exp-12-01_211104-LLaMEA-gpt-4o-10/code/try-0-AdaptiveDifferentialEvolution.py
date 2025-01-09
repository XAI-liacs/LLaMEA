import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim, pop_size=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        # Initialize control parameters
        F = 0.5  # Mutation factor
        CR = 0.7  # Crossover probability

        evaluations = self.pop_size

        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + F * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = (np.random.rand(self.dim) < CR)
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover_mask, mutant, population[i])

                # Evaluate trial vector
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial

                if evaluations >= self.budget:
                    break

                # Adaptation of F and CR
                if evaluations % (self.pop_size * 5) == 0:  # Adaptation every few generations
                    successful_trials = fitness < trial_fitness
                    if np.any(successful_trials):
                        F = np.clip(F + 0.1 * (np.mean(successful_trials) - 0.5), 0.1, 0.9)
                        CR = np.clip(CR + 0.1 * (np.mean(successful_trials) - 0.5), 0.1, 0.9)

        return best_solution, best_fitness
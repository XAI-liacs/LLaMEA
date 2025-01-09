import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None
        self.pop_size = int(budget / dim)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.adaptation_rate = 0.1

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size

        # Store the best solution
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]

        while eval_count < self.budget:
            for i in range(self.pop_size):
                # Select three distinct random indices
                indices = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                a, b, c = population[indices]

                # Mutate and recombine
                mutant = a + self.mutation_factor * (b - c)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, population[i])

                # Evaluate trial individual
                f_trial = func(trial)
                eval_count += 1

                # Selection
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial

                    # If this is the best solution so far
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                # Adapt mutation factor and crossover rate
                self.mutation_factor = 0.5 + self.adaptation_rate * (self.f_opt - f_trial) / (self.f_opt + np.abs(fitness[i] - f_trial))
                self.crossover_rate = 0.9 - self.adaptation_rate * (self.f_opt - f_trial) / (self.f_opt - fitness[i])

                if eval_count >= self.budget:
                    break

        return self.f_opt, self.x_opt
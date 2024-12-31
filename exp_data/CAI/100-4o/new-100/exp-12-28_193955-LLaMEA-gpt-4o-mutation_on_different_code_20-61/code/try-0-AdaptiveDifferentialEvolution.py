import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.cr = 0.9  # Crossover rate
        self.f = 0.5   # Differential weight

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        # Main optimization loop
        while eval_count < self.budget:
            for i in range(self.population_size):
                # Select three distinct individuals
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Mutation: create a donor vector
                mutant = np.clip(a + self.f * (b - c), func.bounds.lb, func.bounds.ub)

                # Crossover: create a trial vector
                crossover_mask = np.random.rand(self.dim) < self.cr
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover_mask, mutant, population[i])

                # Selection: replace if the trial is better
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Update the best found solution
                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial

                # Exit if budget is exhausted
                if eval_count >= self.budget:
                    break

        return self.f_opt, self.x_opt
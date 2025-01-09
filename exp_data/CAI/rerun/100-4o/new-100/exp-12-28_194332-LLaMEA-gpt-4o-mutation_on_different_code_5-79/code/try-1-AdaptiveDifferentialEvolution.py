import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 20  # Starting population size
        self.F = 0.5  # Mutation factor
        self.CR = 0.9  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size

        while num_evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + self.F * (x1 - x2), func.bounds.lb, func.bounds.ub)

                # Crossover
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                num_evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

                if num_evaluations >= self.budget:
                    break

            # Adapt F and CR dynamically
            self.F = np.clip(0.5 + 0.1 * np.random.randn(), 0, 1)
            self.CR = np.clip(0.9 + 0.1 * np.random.randn(), 0, 1)
            self.population_size = max(5, int(self.population_size * 0.95))  # Adjust population size

        return self.f_opt, self.x_opt
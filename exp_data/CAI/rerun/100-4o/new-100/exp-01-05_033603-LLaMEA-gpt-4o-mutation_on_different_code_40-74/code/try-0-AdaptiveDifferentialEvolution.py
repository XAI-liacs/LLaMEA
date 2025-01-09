import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= self.population_size

        # Set initial parameters
        F = 0.5  # Initial mutation factor
        CR = 0.5  # Initial crossover probability

        while self.budget > 0:
            for i in range(self.population_size):
                # Mutation (DE/rand/1)
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                self.budget -= 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update the best solution found
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

                # Adapt F and CR
                if trial_fitness < self.f_opt:
                    F = np.random.normal(loc=0.5, scale=0.1)
                    CR = np.random.normal(loc=0.5, scale=0.1)
                    F = np.clip(F, 0.1, 0.9)
                    CR = np.clip(CR, 0.1, 0.9)

        return self.f_opt, self.x_opt
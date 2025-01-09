import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * self.dim
        self.F = 0.5  # Mutation factor
        self.CR = 0.9  # Crossover probability
        self.elite_fraction = 0.1

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size
        
        while evals < self.budget:
            for i in range(self.population_size):
                # Select indices for mutation
                elite_count = int(self.elite_fraction * self.population_size)
                elite_indices = np.argsort(fitness)[:elite_count]
                indices = np.random.choice(elite_indices, 2, replace=False)
                x0, x1 = population[elite_indices[0]], population[indices[0]]
                x2 = population[np.random.choice(self.population_size)]

                # Mutation with elite preservation
                mutant = np.clip(x0 + self.F * (x1 - x2), func.bounds.lb, func.bounds.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                evals += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

                # Update the best solution found
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                # Adjust mutation factor and crossover rate
                if evals % 100 == 0:
                    improvement_rate = min(fitness) / (self.f_opt + 1e-9)
                    self.F = 0.4 + 0.6 * improvement_rate
                    self.CR = 0.8 - 0.3 * improvement_rate
            
        return self.f_opt, self.x_opt
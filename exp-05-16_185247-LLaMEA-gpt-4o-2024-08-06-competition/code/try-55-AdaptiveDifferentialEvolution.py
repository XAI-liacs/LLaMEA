import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.bounds = (-100.0, 100.0)

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        evaluations = self.population_size
        f_opt_old = self.f_opt

        while evaluations < self.budget:
            # Calculate diversity for adaptive mutation
            diversity = np.mean(np.std(population, axis=0))
            F = 0.5 + 0.5 * (diversity / (self.bounds[1] - self.bounds[0])) * (self.f_opt / (np.mean(fitness) + 1e-9))
            relative_improvement = (f_opt_old - self.f_opt) / (abs(f_opt_old) + 1e-9)  # New line
            CR = 0.5 + 0.4 * (1 - relative_improvement)  # Changed line
            new_population = np.empty_like(population)

            for i in range(self.population_size):
                # Select three distinct random individuals
                idxs = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                x1, x2, x3 = population[idxs]

                # Mutation
                mutant = np.clip(x1 + F * (x2 - x3), self.bounds[0], self.bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        f_opt_old = self.f_opt
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    new_population[i] = population[i]

                evaluations += 1
                if evaluations >= self.budget:
                    break

            population = new_population

        return self.f_opt, self.x_opt
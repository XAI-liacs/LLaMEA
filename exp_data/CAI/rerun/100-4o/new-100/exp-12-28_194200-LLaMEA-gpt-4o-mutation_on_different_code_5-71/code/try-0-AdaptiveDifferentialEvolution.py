import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=100):
        self.budget = budget
        self.dim = dim
        self.pop_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5.0, 5.0)
        self.F_mean = 0.5
        self.CR_mean = 0.5

    def __call__(self, func):
        lb, ub = self.bounds
        # Initialize population
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        n_evals = self.pop_size
        while n_evals < self.budget:
            F = np.clip(np.random.normal(self.F_mean, 0.1), 0, 1)
            CR = np.clip(np.random.normal(self.CR_mean, 0.1), 0, 1)
            new_population = np.copy(population)

            # Generate new candidates
            for i in range(self.pop_size):
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                
                mutant = np.clip(x1 + F * (x2 - x3), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                f = func(trial)
                n_evals += 1
                
                if f < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = f

            population = new_population

            # Update mean F and CR based on successful trials
            if np.any(fitness < self.f_opt):
                improved = fitness < self.f_opt
                self.F_mean = (self.F_mean + F * np.sum(improved)) / (1 + np.sum(improved))
                self.CR_mean = (self.CR_mean + CR * np.sum(improved)) / (1 + np.sum(improved))

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.f_opt:
                self.f_opt = fitness[min_idx]
                self.x_opt = population[min_idx]

        return self.f_opt, self.x_opt
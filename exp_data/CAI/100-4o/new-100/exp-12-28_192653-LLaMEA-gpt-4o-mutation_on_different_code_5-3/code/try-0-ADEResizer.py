import numpy as np

class ADEResizer:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_individuals = 4 * dim  # Initial population size
        self.F = 0.7  # Differential weight
        self.CR = 0.5  # Crossover probability
        self.success_rate = 0.2  # Desired success rate
        self.fail_rate = 0.1  # Desired fail rate
        
    def adapt_parameters(self, success, fail):
        if success:
            self.F = min(1.0, self.F + 0.1)
            self.CR = min(1.0, self.CR + 0.1)
        if fail:
            self.F = max(0.1, self.F - 0.1)
            self.CR = max(0.1, self.CR - 0.1)

    def resize_population(self, success):
        if success:
            self.num_individuals = min(100, self.num_individuals + 1)
        else:
            self.num_individuals = max(4 * self.dim, self.num_individuals - 1)

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_individuals, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        evaluations = self.num_individuals
        while evaluations < self.budget:
            for i in range(self.num_individuals):
                indices = np.random.choice(np.delete(np.arange(self.num_individuals), i), 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + self.F * (x1 - x2), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    success = True
                else:
                    success = False
                
                self.adapt_parameters(success, not success)
                self.resize_population(success)
            
            if evaluations >= self.budget:
                break

            if fitness.min() < self.f_opt:
                self.f_opt = fitness.min()
                self.x_opt = population[fitness.argmin()]

        return self.f_opt, self.x_opt
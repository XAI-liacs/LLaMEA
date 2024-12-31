import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.bounds = (-5.0, 5.0)
        self.F = 0.5  # Mutation factor
        self.CR = 0.9  # Crossover rate

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_spent = self.population_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]

        while budget_spent < self.budget:
            new_population = np.copy(population)
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice([idx for idx in range(self.population_size) if idx != i], 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + self.F * (x2 - x3), self.bounds[0], self.bounds[1])
                
                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                budget_spent += 1
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                # Adaptive parameters
                if budget_spent % (2 * self.population_size) == 0:
                    self.F = np.random.uniform(0.4, 0.9)
                    self.CR = np.random.uniform(0.4, 0.9)

                if budget_spent >= self.budget:
                    break

            population = new_population

        return self.f_opt, self.x_opt
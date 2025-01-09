import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        evaluations = self.population_size
        
        # Adaptive parameters
        F = 0.5
        CR = 0.9
        tau1 = 0.1
        tau2 = 0.1

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + F * (x2 - x3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                # Crossover
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

            # Adapt F and CR based on diversity
            if np.random.rand() < tau1:
                F = np.random.uniform(0.1, 0.9)
            if np.random.rand() < tau2:
                CR = np.random.uniform(0.1, 0.9)

        return self.f_opt, self.x_opt
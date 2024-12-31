import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None
        self.population_size = 10 * self.dim
        self.mutation_factor = 0.5
        self.crossover_probability = 0.7
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        lb, ub = self.bounds
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                idxs = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[idxs]
                mutant = np.clip(x1 + self.mutation_factor * (x2 - x3), lb, ub)
                crossover = np.random.rand(self.dim) < self.crossover_probability
                trial = np.where(crossover, mutant, population[i])
                
                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                if evaluations >= self.budget:
                    break
            
            mean_fitness = np.mean(fitness)
            self.mutation_factor = 0.5 + (0.5 * (self.f_opt / mean_fitness))
            self.crossover_probability = 0.3 + (0.4 * (mean_fitness / self.f_opt))

        return self.f_opt, self.x_opt
import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 10 * dim  # Common heuristic for population size
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.adapt_rate = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget - self.pop_size):
            for i in range(self.pop_size):
                # Mutation: select three distinct individuals
                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt, self.x_opt = trial_fitness, trial
                
                # Adaptive parameters
                self.mutation_factor += self.adapt_rate * (np.random.rand() - 0.5)
                self.crossover_rate += self.adapt_rate * (np.random.rand() - 0.5)
                self.mutation_factor = np.clip(self.mutation_factor, 0.1, 0.9)
                self.crossover_rate = np.clip(self.crossover_rate, 0.1, 0.9)

        return self.f_opt, self.x_opt
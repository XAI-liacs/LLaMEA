import numpy as np

class DynamicDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        
        # Initialize population
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= self.population_size

        # Evolutionary loop
        while self.budget > 0:
            for i in range(self.population_size):
                if self.budget <= 0:
                    break

                # Mutation: Generate mutant vector
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                adapt_m_factor = 0.5 + np.abs((fitness[indices[0]] - fitness[indices[1]]) / max(fitness))
                mutant = np.clip(a + adapt_m_factor * (b - c), bounds[0], bounds[1])

                # Crossover: Generate trial vector
                dynamic_crossover = self.crossover_rate + 0.1 * np.random.rand()
                crossover_mask = np.random.rand(self.dim) < dynamic_crossover
                trial = np.where(crossover_mask, mutant, population[i])

                # Selection
                f_trial = func(trial)
                self.budget -= 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

                # Track the best solution
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

        return self.f_opt, self.x_opt
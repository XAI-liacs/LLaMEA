import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, size=(self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        while evals < self.budget:
            # Self-adaptive population size
            if evals % (self.budget // 10) == 0:
                self.population_size = np.clip(self.population_size + np.random.randint(-5, 6), 5, 20 * self.dim)
                population = population[:self.population_size]
                fitness = fitness[:self.population_size]
            
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 5, replace=False)
                x1, x2, x3, x4, x5 = population[indices]
                # New mutation strategy using five vectors
                mutant = np.clip(x1 + self.mutation_factor * (x2 - x3 + x4 - x5), lb, ub)

                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover_mask, mutant, population[i])

                f_trial = func(trial)
                evals += 1

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if evals >= self.budget:
                    break

            if np.random.rand() < 0.1:
                self.mutation_factor = np.random.uniform(0.4, 0.9)
                self.crossover_rate = np.random.uniform(0.5, 0.9)

        return self.f_opt, self.x_opt
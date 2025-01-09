import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size_factor=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size_factor * dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= self.pop_size

        while self.budget > 0:
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds[0], bounds[1])
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[i])
                
                f_trial = func(trial)
                self.budget -= 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                if self.budget <= 0:
                    break

            # Update F and CR based on success rates (improving balance)
            success_rate = np.mean(fitness < fitness.mean())
            self.F = np.clip(self.F * (1 + 0.1 * (success_rate - 0.5)), 0, 1)
            self.CR = np.clip(self.CR * (1 + 0.1 * (success_rate - 0.5)), 0, 1)

            # Reduce population size as the budget decreases
            if self.budget < self.pop_size:
                survivors = np.argsort(fitness)[:self.budget]
                population = population[survivors]
                fitness = fitness[survivors]
                self.pop_size = len(survivors)

        return self.f_opt, self.x_opt
import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.CR = 0.5  # Crossover probability
        self.F = 0.8   # Mutation factor
        self.success_rates = []  # To adjust CR and F

    def adapt_parameters(self):
        if self.success_rates:
            mean_success_rate = np.mean(self.success_rates)
            self.CR = 0.9 * self.CR + 0.1 * mean_success_rate
            self.F = 0.9 * self.F + 0.1 * (1 - mean_success_rate)

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(low=bounds[0], high=bounds[1], size=(self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.f_opt = fitness.min()
        self.x_opt = population[fitness.argmin()]

        evaluations = self.pop_size
        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + self.F * (x2 - x3), bounds[0], bounds[1])

                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial
                    self.success_rates.append(1)  # Successful trial
                else:
                    self.success_rates.append(0)  # Unsuccessful trial

                # Update best known solution
                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial

                if evaluations >= self.budget:
                    break

            # Adapt parameters based on success history
            self.adapt_parameters()
            self.success_rates = []  # Reset for the next generation

        return self.f_opt, self.x_opt
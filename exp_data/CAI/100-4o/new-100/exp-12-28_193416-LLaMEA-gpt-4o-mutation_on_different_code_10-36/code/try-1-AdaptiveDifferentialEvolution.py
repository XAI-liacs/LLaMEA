import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.initial_F = F
        self.initial_CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        # Update the optimal solution
        self.update_optimal(population, fitness)

        eval_count = self.pop_size
        iteration = 0

        while eval_count < self.budget:
            new_population = np.zeros_like(population)

            # Dynamic adaptation of F and CR
            self.F = self.initial_F * (1.0 - (eval_count / self.budget))
            self.CR = self.initial_CR * (1.0 - (eval_count / self.budget))

            for i in range(self.pop_size):
                # Mutation: Select three random vectors
                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = a + self.F * (b - c)
                np.clip(mutant, func.bounds.lb, func.bounds.ub, out=mutant)

                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i]

                # Global best update
                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial

                if eval_count >= self.budget:
                    break

            population = new_population
            iteration += 1

        return self.f_opt, self.x_opt

    def update_optimal(self, population, fitness):
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.f_opt:
            self.f_opt = fitness[min_idx]
            self.x_opt = population[min_idx]
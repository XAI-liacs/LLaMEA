import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 5 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.gaussian_mutation_prob = 0.1  # Probability to apply Gaussian mutation

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        while evals < self.budget:
            for i in range(self.population_size):
                if evals >= self.budget:
                    break
                # Mutation step
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(population[a] + self.F * (population[b] - population[c]), func.bounds.lb, func.bounds.ub)

                # Crossover step
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])

                # Gaussian Mutation
                if np.random.rand() < self.gaussian_mutation_prob:
                    trial += np.random.normal(0, 0.1, self.dim)
                    trial = np.clip(trial, func.bounds.lb, func.bounds.ub)

                # Selection step
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Update the best solution found
                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial

        return self.f_opt, self.x_opt
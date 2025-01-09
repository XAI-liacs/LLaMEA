import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.5  # Mutation factor
        self.CR = 0.9  # Crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, 
                                            (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.best_solution = None
        self.best_fitness = np.inf

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            for i in range(self.population_size):
                if evals >= self.budget:
                    break

                # Mutation and crossover
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, self.population[i])

                # Evaluate trial solution
                trial_fitness = func(trial)
                evals += 1

                # Greedy selection
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                    # Update best solution
                    if trial_fitness < self.best_fitness:
                        self.best_solution = trial
                        self.best_fitness = trial_fitness

            # Dynamic adjustment of F and CR
            self.F = np.clip(np.random.normal(0.5, 0.1), 0, 1)
            self.CR = np.clip(np.random.normal(0.9, 0.1), 0, 1)

        return self.best_solution
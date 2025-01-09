import numpy as np

class MemeticDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def local_search(self, x, func):
        # A simple adaptive local search using gradient approximation
        step_size = 0.1
        best_x = x
        best_f = func(x)
        for _ in range(10):
            perturbation = np.random.uniform(-step_size, step_size, size=self.dim)
            new_x = np.clip(x + perturbation, func.bounds.lb, func.bounds.ub)
            new_f = func(new_x)
            if new_f < best_f:
                best_x, best_f = new_x, new_f
        return best_x, best_f

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Mutation
                indices = np.random.choice([idx for idx in range(self.population_size) if idx != i], 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + self.F * (x2 - x3), func.bounds.lb, func.bounds.ub)

                # Crossover
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])

                # Selection
                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

                    # Local search phase
                    trial, f_trial = self.local_search(trial, func)
                    evaluations += 10  # Assume 10 evaluations for local search

                    if f_trial < fitness[i]:
                        population[i] = trial
                        fitness[i] = f_trial

                # Update global best
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

        return self.f_opt, self.x_opt
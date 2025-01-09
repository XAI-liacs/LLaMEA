import numpy as np

class DEWithLocalSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def local_search(self, x, func):
        step_size = 0.05 * (func.bounds.ub - func.bounds.lb)
        for _ in range(5):  # Perform a few local steps
            neighbor = x + np.random.uniform(-step_size, step_size)
            np.clip(neighbor, func.bounds.lb, func.bounds.ub, out=neighbor)
            f_neighbor = func(neighbor)
            if f_neighbor < self.f_opt:
                self.f_opt = f_neighbor
                self.x_opt = neighbor

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget - self.population_size):
            diversity = np.mean(np.std(population, axis=0))  # Calculate population diversity
            for i in range(self.population_size):
                # Mutation
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                self.F = 0.5 + 0.5 * diversity  # Adaptive differential weight
                mutant = population[a] + self.F * (population[b] - population[c])
                np.clip(mutant, func.bounds.lb, func.bounds.ub, out=mutant)

                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])

                # Selection
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        self.local_search(self.x_opt, func)

        return self.f_opt, self.x_opt
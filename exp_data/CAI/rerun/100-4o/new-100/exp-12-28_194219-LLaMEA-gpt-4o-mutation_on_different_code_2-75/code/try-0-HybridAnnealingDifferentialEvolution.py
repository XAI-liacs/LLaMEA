import numpy as np

class HybridAnnealingDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.temperature = 1.0
        self.f_opt = np.Inf
        self.x_opt = None

    def simulated_annealing(self, current, candidate, func):
        diff = func(candidate) - func(current)
        if diff < 0 or np.random.rand() < np.exp(-diff / self.temperature):
            return candidate
        return current

    def differential_evolution(self, population, func):
        new_population = np.copy(population)
        for i in range(self.population_size):
            candidates = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = population[candidates]
            mutant = np.clip(a + 0.8 * (b - c), -5.0, 5.0)
            trial = np.where(np.random.rand(self.dim) < 0.9, mutant, population[i])
            new_population[i] = self.simulated_annealing(population[i], trial, func)
        return new_population

    def __call__(self, func):
        func.bounds = lambda: None
        func.bounds.lb = -5.0
        func.bounds.ub = 5.0

        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        for _ in range(self.budget // self.population_size):
            population = self.differential_evolution(population, func)
            for ind in population:
                f = func(ind)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = ind
            self.temperature *= 0.99  # Annealing schedule

        return self.f_opt, self.x_opt
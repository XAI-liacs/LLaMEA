import numpy as np

class HybridDifferentialEvolutionLocalSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability

    def initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))

    def differential_evolution(self, population, func):
        new_population = np.copy(population)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            if func(trial) < func(population[i]):
                new_population[i] = trial
        return new_population

    def local_search(self, x, func):
        for _ in range(5):  # Perform 5 local search steps
            step = np.random.uniform(-0.1, 0.1, self.dim)
            neighbor = np.clip(x + step, func.bounds.lb, func.bounds.ub)
            if func(neighbor) < func(x):
                x = neighbor
        return x

    def __call__(self, func):
        population = self.initialize_population(func.bounds)
        evaluations = self.population_size
        
        while evaluations < self.budget:
            population = self.differential_evolution(population, func)
            evaluations += self.population_size
            if evaluations + self.population_size > self.budget:
                break
            
            # Apply local search on each individual
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                population[i] = self.local_search(population[i], func)
                evaluations += 1

            # Update best solution
            for x in population:
                f = func(x)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = x

        return self.f_opt, self.x_opt
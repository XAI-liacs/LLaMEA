import numpy as np

class HybridLevyDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = int(budget / dim)
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')

    def levy_flight(self, beta=1.5):
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / abs(v)**(1 / beta)
        return step

    def differential_evolution(self, func, lb, ub):
        F_init = 0.5  # Initial mutation factor
        CR = 0.9  # Crossover probability

        for _ in range(self.budget):
            for i in range(self.population_size):
                # Adjust mutation factor based on current convergence speed
                F = F_init + 0.1 * ((self.best_fitness - func(self.population[i])) / abs(self.best_fitness))

                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                x = self.population[i]
                mutant = np.clip(self.population[a] + F * (self.population[b] - self.population[c]), lb, ub)
                
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, x)
                fitness = func(trial)

                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = trial
                
                if fitness < func(x):
                    self.population[i] = trial

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        self.differential_evolution(func, lb, ub)

        for _ in range(self.budget // self.population_size):
            for i in range(self.population_size):
                step = self.levy_flight()
                candidate = np.clip(self.population[i] + step, lb, ub)
                fitness = func(candidate)
                
                if fitness < func(self.population[i]):
                    self.population[i] = candidate

                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = candidate

        return self.best_solution
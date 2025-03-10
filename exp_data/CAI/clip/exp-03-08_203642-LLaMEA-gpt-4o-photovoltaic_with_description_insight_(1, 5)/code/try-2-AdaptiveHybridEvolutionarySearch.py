import numpy as np
from scipy.optimize import minimize

class AdaptiveHybridEvolutionarySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.f = 0.8  # DE mutation factor
        self.cr = 0.9  # DE crossover rate
        self.evaluations = 0

    def mutate(self, population, target_idx):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = population[a] + self.f * (population[b] - population[c]) + 0.1 * (population[a] - population[c])
        return np.clip(mutant, self.lb, self.ub)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.cr
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def local_search(self, x):
        res = minimize(self.func, x, bounds=self.bounds, method='L-BFGS-B')
        return res.x if res.success else x

    def evolutionary_search_step(self, population):
        new_population = np.copy(population)
        for i in range(self.population_size):
            mutant = self.mutate(population, i)
            trial = self.crossover(population[i], mutant)
            trial = self.local_search(trial)  # Apply local search
            if self.func(trial) < self.func(population[i]):
                new_population[i] = trial
        return new_population

    def __call__(self, func):
        self.func = func
        self.bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        best_solution = population[np.argmin([self.func(ind) for ind in population])]

        while self.evaluations < self.budget:
            population = self.evolutionary_search_step(population)
            best_candidate = population[np.argmin([self.func(ind) for ind in population])]
            if self.func(best_candidate) < self.func(best_solution):
                best_solution = best_candidate

            self.evaluations += self.population_size
            # Optional: Reduce dimension by initializing fewer layers at first
            active_layers = min(self.dim, int((self.evaluations / self.budget) * self.dim))
            self.dim = active_layers

        return best_solution
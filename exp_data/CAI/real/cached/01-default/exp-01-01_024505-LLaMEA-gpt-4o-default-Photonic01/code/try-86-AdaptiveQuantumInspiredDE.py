import numpy as np

class AdaptiveQuantumInspiredDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, min(60, budget // 5))
        self.population = None
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.best_solution = None
        self.best_fitness = float('inf')
        self.stochastic_local_search_prob = 0.25
        self.levy_prob = 0.3

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        for i, f in enumerate(fitness):
            if f < self.best_fitness:
                self.best_fitness = f
                self.best_solution = self.population[i]
        return fitness

    def mutate(self, target_index, lb, ub):
        indices = list(range(self.population_size))
        indices.remove(target_index)
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant = a + self.F * (b - c)
        mutant = np.clip(mutant, lb, ub)
        return mutant

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def levy_flight(self, i, lb, ub):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        step = u / np.abs(v) ** (1 / beta)
        return np.clip(self.population[i] + 0.01 * step * (self.population[i] - self.best_solution), lb, ub)

    def stochastic_local_search(self, ind, lb, ub):
        perturbation = 0.01 * (ub - lb) * np.random.randn(self.dim)
        return np.clip(ind + perturbation, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            fitness = self.evaluate_population(func)
            evaluations += self.population_size

            if evaluations >= self.budget:
                break

            for i in range(self.population_size):
                mutant = self.mutate(i, lb, ub)
                trial = self.crossover(self.population[i], mutant)

                if np.random.rand() < self.levy_prob:
                    trial = self.levy_flight(i, lb, ub)

                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    self.population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial

                if evaluations >= self.budget:
                    break

                if np.random.rand() < self.stochastic_local_search_prob:
                    self.population[i] = self.stochastic_local_search(self.population[i], lb, ub)

        return self.best_solution, self.best_fitness
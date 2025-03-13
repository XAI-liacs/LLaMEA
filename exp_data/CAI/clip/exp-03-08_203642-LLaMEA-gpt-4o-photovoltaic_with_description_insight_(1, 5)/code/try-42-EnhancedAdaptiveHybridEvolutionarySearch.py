import numpy as np
from scipy.optimize import minimize

class EnhancedAdaptiveHybridEvolutionarySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.f = 0.8
        self.cr = 0.9
        self.evaluations = 0

    def adapt_parameters(self):
        self.f = 0.5 + np.random.rand() * 0.5 * (1 - self.evaluations / self.budget)
        self.cr = 0.8 + np.random.rand() * 0.2

    def mutate(self, population, target_idx):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = population[a] + self.f * (population[b] - population[c])
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

    def detect_and_preserve_modular_structures(self, population):
        modular_population = []
        for individual in population:
            if np.random.rand() < 0.5:
                modular_population.append(self.robustness_check(individual))
            else:
                modular_population.append(individual)
        return modular_population

    def quantum_inspired_mutation(self, solution):
        q_mutation = np.random.normal(0, 0.1 * (1 - self.evaluations / self.budget), size=solution.shape)
        return np.clip(solution + q_mutation, self.lb, self.ub)

    def evolutionary_search_step(self, population):
        new_population = np.copy(population)
        for i in range(self.population_size):
            self.adapt_parameters()
            mutant = self.mutate(population, i)
            trial = self.crossover(population[i], mutant)
            trial = self.local_search(trial)
            trial = self.quantum_inspired_mutation(trial)
            if self.func(trial) < self.func(population[i]):
                new_population[i] = trial
        return self.detect_and_preserve_modular_structures(new_population)

    def __call__(self, func):
        self.func = func
        self.bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub

        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        best_solution = population[np.argmin([self.func(ind) for ind in population])]

        while self.evaluations < self.budget:
            self.population_size = max(10, int(20 * (1 - self.evaluations / self.budget)))
            population = self.evolutionary_search_step(population)
            best_candidate = population[np.argmin([self.func(ind) for ind in population])]
            if self.func(best_candidate) < self.func(best_solution):
                best_solution = best_candidate

            self.evaluations += self.population_size
            active_layers = min(self.dim, int((self.evaluations / self.budget) * self.dim)) + 1
            if self.evaluations % 10 == 0: self.f *= 0.95

        return best_solution
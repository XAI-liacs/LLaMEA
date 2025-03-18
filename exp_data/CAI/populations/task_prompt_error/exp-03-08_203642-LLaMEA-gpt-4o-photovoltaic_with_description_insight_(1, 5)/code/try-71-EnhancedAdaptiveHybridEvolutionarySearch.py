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
        self.learning_rate = 0.1
        self.f_memory = 0.8

    def adapt_parameters(self):
        self.f = self.f_memory + np.random.rand() * 0.3 * (1 - self.evaluations / self.budget)
        self.cr = 0.9 + np.random.rand() * 0.1

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

    def robustness_check(self, solution):
        perturbation_scale = 0.01 * (1 - self.evaluations / self.budget)
        perturbation = np.random.normal(0, perturbation_scale, size=solution.shape)
        robust_solution = np.clip(solution + perturbation, self.lb, self.ub)
        return robust_solution

    def quantum_inspired_mutation(self, solution):
        q_mutation = np.random.normal(0, 0.05 * (1 - self.evaluations / self.budget), size=solution.shape)
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
                self.f_memory = 0.9 * self.f_memory + 0.1 * self.f
            else:
                new_population[i] = population[i] + self.learning_rate * (trial - population[i])
                if np.random.rand() < 0.05:  # New line (1)
                    new_population[i] = np.random.uniform(self.lb, self.ub, self.dim)  # New line (2)
        return new_population

    def __call__(self, func):
        self.func = func
        self.bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub

        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        best_solution = population[np.argmin([self.func(ind) for ind in population])]

        while self.evaluations < self.budget:
            self.population_size = max(15, int(25 * (1 - self.evaluations**0.5 / self.budget**0.5)))
            population = self.evolutionary_search_step(population)
            best_candidate = population[np.argmin([self.func(ind) for ind in population])]
            if self.func(best_candidate) < self.func(best_solution):
                best_solution = best_candidate
                self.learning_rate = 0.1 + 0.5 * (1 - self.evaluations / self.budget)

            self.evaluations += self.population_size
            active_layers = min(self.dim, int((self.evaluations / self.budget) * self.dim)) + 1
            if self.evaluations % 10 == 0:  # Changed condition (3)
                self.f *= 0.92  # Changed line (4)

        return best_solution
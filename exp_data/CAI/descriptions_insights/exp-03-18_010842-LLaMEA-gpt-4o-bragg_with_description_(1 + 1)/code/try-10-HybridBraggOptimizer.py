import numpy as np

class HybridBraggOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.base_population_size = 10 * dim  # Base population size is scaled with dimension
        self.adaptive_population_size = self.base_population_size  # Start with base size
        self.cr = 0.9  # Increased crossover probability for better diversity
        self.f = 0.8  # Differential weight
        self.local_search_radius = 0.1  # Reduced radius for more granular local search exploitation
        self.population = None
        self.best_solution = None
        self.best_score = float('-inf')

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.adaptive_population_size, self.dim))

    def evaluate_population(self, func):
        scores = np.array([func(ind) for ind in self.population])
        best_idx = np.argmax(scores)
        if scores[best_idx] > self.best_score:
            self.best_score = scores[best_idx]
            self.best_solution = self.population[best_idx].copy()

    def differential_evolution_step(self, bounds, func):
        new_population = np.zeros_like(self.population)
        for i in range(self.adaptive_population_size):
            idxs = [idx for idx in range(self.adaptive_population_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            self.f = np.random.uniform(0.4, 1.0)  # Broadened range for adaptive F
            mutant = np.clip(a + self.f * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < self.cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.population[i])
            if func(trial) > func(self.population[i]):
                new_population[i] = trial
            else:
                new_population[i] = self.population[i]
        self.population = new_population

    def local_search(self, bounds, func):
        for i in range(self.adaptive_population_size):
            candidate = self.population[i] + np.random.uniform(-self.local_search_radius, self.local_search_radius, self.dim)
            candidate = np.clip(candidate, bounds.lb, bounds.ub)
            if func(candidate) > func(self.population[i]):
                self.population[i] = candidate

    def multi_strategy_local_search(self, bounds, func):
        for i in range(self.adaptive_population_size):
            perturb = np.random.uniform(-self.local_search_radius, self.local_search_radius, self.dim)
            candidate = np.clip(self.population[i] + perturb, bounds.lb, bounds.ub)
            exploration_candidate = np.clip(self.population[i] + 2 * perturb, bounds.lb, bounds.ub)
            if func(candidate) > func(self.population[i]):
                self.population[i] = candidate
            elif func(exploration_candidate) > func(self.population[i]):
                self.population[i] = exploration_candidate

    def __call__(self, func):
        func_counter = 0
        bounds = func.bounds
        self.initialize_population(bounds)
        self.evaluate_population(func)
        func_counter += self.adaptive_population_size

        while func_counter < self.budget:
            if func_counter % 3 == 0:
                self.hybrid_mutation(bounds)
            else:
                self.differential_evolution_step(bounds, func)
            func_counter += self.adaptive_population_size
            if func_counter < self.budget:
                self.multi_strategy_local_search(bounds, func)
                func_counter += self.adaptive_population_size
            self.evaluate_population(func)

        return self.best_solution
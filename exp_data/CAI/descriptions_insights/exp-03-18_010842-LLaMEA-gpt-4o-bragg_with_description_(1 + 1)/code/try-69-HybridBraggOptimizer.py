import numpy as np

class HybridBraggOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Heuristic to determine population size
        self.cr = 0.8  # Crossover probability
        self.f = 0.8  # Differential weight
        self.local_search_radius = 0.3  # Increased initial radius for local search exploitation
        self.population = None
        self.best_solution = None
        self.best_score = float('-inf')

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))

    def evaluate_population(self, func):
        scores = np.array([func(ind) for ind in self.population])
        best_idx = np.argmax(scores)
        if scores[best_idx] > self.best_score:
            self.best_score = scores[best_idx]
            self.best_solution = self.population[best_idx].copy()

    def adaptive_differential_evolution_step(self, bounds, func):
        new_population = np.zeros_like(self.population)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            self.f = 0.5 + np.random.rand() * 0.5  # Adaptive F
            self.cr = 0.6 + np.random.rand() * 0.4  # Dynamic crossover probability
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
        search_radius_factor = np.linspace(0.95, 0.85, self.population_size)  # Adjusted range
        for i, radius_factor in enumerate(search_radius_factor):
            candidate = self.population[i] + np.random.uniform(-self.local_search_radius * radius_factor, 
                                                              self.local_search_radius * radius_factor, self.dim)
            candidate = np.clip(candidate, bounds.lb, bounds.ub)
            if func(candidate) > func(self.population[i]):
                self.population[i] = candidate

    def hybrid_mutation(self, bounds):
        for i in range(self.population_size):
            idxs = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = self.population[idxs]
            new_candidate = np.clip(a + np.random.uniform(0.5, 1.0) * (b - c), bounds.lb, bounds.ub)
            self.population[i] = new_candidate

    def convergence_acceleration(self, func):
        acceleration_factor = 0.06  # Slight increase in acceleration factor
        for i in range(self.population_size):
            if np.random.random() < acceleration_factor:
                perturbation = np.random.uniform(-0.1, 0.1, self.dim)
                candidate = self.population[i] + perturbation
                candidate = np.clip(candidate, func.bounds.lb, func.bounds.ub)
                if func(candidate) > func(self.population[i]):
                    self.population[i] = candidate

    def __call__(self, func):
        func_counter = 0
        bounds = func.bounds
        self.initialize_population(bounds)
        self.evaluate_population(func)
        func_counter += self.population_size

        while func_counter < self.budget:
            if func_counter % 3 == 0:
                self.hybrid_mutation(bounds)
            else:
                self.adaptive_differential_evolution_step(bounds, func)
            func_counter += self.population_size
            if func_counter < self.budget:
                self.local_search_radius *= 0.99  # Dynamic adjustment
                self.local_search(bounds, func)
                func_counter += self.population_size
                self.convergence_acceleration(func)
            self.evaluate_population(func)

        return self.best_solution
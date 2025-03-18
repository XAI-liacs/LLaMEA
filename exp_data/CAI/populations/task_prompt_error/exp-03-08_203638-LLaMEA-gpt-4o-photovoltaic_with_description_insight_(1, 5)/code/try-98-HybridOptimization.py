import numpy as np
from scipy.optimize import minimize

class HybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.population_size = 25  # Adjusted for better exploration
        self.num_layers = 10
        self.increment_step = max(1, self.dim // 8)  # Adjusted step for layer increments
        self.best_solution = None
        self.best_score = -np.inf
        self.bounds = None
        self.noise_tolerance = 0.01
        self.layer_specific_mutation = True

    def _initialize_population(self):
        return np.random.uniform(self.bounds.lb, self.bounds.ub, (self.population_size, self.dim))

    def _evaluate(self, individual, func):
        if self.evaluations >= self.budget:
            return np.inf
        self.evaluations += 1
        score = func(individual)
        noise_adjusted_score = score + np.random.normal(0, self.noise_tolerance)
        return noise_adjusted_score

    def _differential_evolution(self, func):
        self.population_size = min(60, 20 + int(30 * (self.evaluations / self.budget)))
        population = self._initialize_population()
        scores = np.array([self._evaluate(ind, func) for ind in population])
        
        elite_idx = np.argmax(scores)
        elite = population[elite_idx].copy()

        for _ in range(self.budget // self.population_size):
            if self.evaluations >= self.budget:
                break
            
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                F = 0.5 + 0.3 * np.sin(np.pi * self.evaluations / self.budget)  
                F *= (1 + 0.3 * (self.best_score / np.max(scores))) # Adjusted mutation scaling factor
                if self.layer_specific_mutation:
                    # Layer-specific scaling
                    layer_factor = 0.8 + 0.4 * np.random.random()
                    F *= layer_factor
                mutant = np.clip(a + F * (b - c), self.bounds.lb, self.bounds.ub)

                diversity_metric = np.std(scores)
                avg_score = np.mean(scores)
                crossover_rate = 0.7 + 0.2 * (1 - diversity_metric / np.max(scores))
                crossover_rate += 0.1 * (self.best_score - avg_score)  # New adaptive crossover rate adjustment
                cross_points = np.random.rand(self.dim) < crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                trial_score = self._evaluate(trial, func)
                
                if trial_score > scores[i] and trial_score > self.best_score - self.noise_tolerance:
                    population[i] = trial
                    scores[i] = trial_score

                if trial_score > self.best_score + self.noise_tolerance:
                    self.best_solution = trial
                    self.best_score = trial_score

            population[elite_idx] = elite
            self.noise_tolerance = 0.005 * (1 + 0.5 * np.cos(np.pi * self.evaluations / self.budget))  # Adjusted noise tolerance

    def _local_search(self, func):
        if self.best_solution is None:
            return

        def local_func(x):
            score = self._evaluate(x, func)
            return -score

        bounds = list(zip(self.bounds.lb, self.bounds.ub))
        res = minimize(local_func, self.best_solution, bounds=bounds, options={'xatol': 1e-4})
        if -res.fun > self.best_score + self.noise_tolerance:
            self.best_solution, self.best_score = res.x, -res.fun

    def _increase_layers(self):
        if np.random.rand() < 0.85:  # Adjusted probability for layer increment
            self.num_layers = min(self.num_layers + self.increment_step, self.dim)

    def __call__(self, func):
        self.bounds = func.bounds
        while self.evaluations < self.budget:
            self._differential_evolution(func)
            self._local_search(func)
            self._increase_layers()

        return self.best_solution
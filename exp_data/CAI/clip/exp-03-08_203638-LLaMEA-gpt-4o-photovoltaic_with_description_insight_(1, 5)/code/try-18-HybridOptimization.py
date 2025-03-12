import numpy as np
from scipy.optimize import minimize

class HybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.population_size = 20
        self.num_layers = 10  # Starting number of layers for gradual complexity increase
        self.increment_step = max(1, self.dim // 10)
        self.best_solution = None
        self.best_score = -np.inf
        self.bounds = None
        self.noise_tolerance = 0.01  # Added noise tolerance threshold

    def _initialize_population(self):
        return np.random.uniform(self.bounds.lb, self.bounds.ub, (self.population_size, self.dim))

    def _evaluate(self, individual, func):
        if self.evaluations >= self.budget:
            return np.inf
        self.evaluations += 1
        score = func(individual)
        return score + np.random.normal(0, self.noise_tolerance)  # Added noise to score

    def _differential_evolution(self, func):
        population = self._initialize_population()
        scores = np.array([self._evaluate(ind, func) for ind in population])

        for _ in range(self.budget // self.population_size):
            if self.evaluations >= self.budget:
                break

            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Adaptive mutation
                F = 0.5 + (0.5 * (self.evaluations / self.budget))
                mutant = np.clip(a + F * (b - c), self.bounds.lb, self.bounds.ub)

                # Adaptive crossover
                crossover_rate = 0.9 - (0.3 * (self.evaluations / self.budget))
                cross_points = np.random.rand(self.dim) < crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection with elitism
                trial_score = self._evaluate(trial, func)
                if trial_score > scores[i]:
                    population[i] = trial
                    scores[i] = trial_score

                # Update best solution
                if trial_score > self.best_score + self.noise_tolerance:  # Adjusted for noise
                    self.best_solution = trial
                    self.best_score = trial_score

    def _local_search(self, func):
        if self.best_solution is None:
            return

        def local_func(x):
            score = self._evaluate(x, func)
            return -score  # Convert to minimization

        bounds = list(zip(self.bounds.lb, self.bounds.ub))
        res = minimize(local_func, self.best_solution, bounds=bounds)
        if -res.fun > self.best_score + self.noise_tolerance:  # Adjusted for noise
            self.best_solution, self.best_score = res.x, -res.fun

    def _increase_layers(self):
        self.num_layers = min(self.num_layers + self.increment_step, self.dim)
        if np.random.rand() < 0.5:  # Dynamic layer adaptation
            self._reinitialize_population()

    def _reinitialize_population(self):
        pass  # Placeholder for potential reinitialization logic

    def __call__(self, func):
        self.bounds = func.bounds
        while self.evaluations < self.budget:
            self._differential_evolution(func)
            self._local_search(func)
            self._increase_layers()

        return self.best_solution
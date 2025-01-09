import numpy as np

class HybridDELocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None
        self.best_value = float('inf')

    def differential_evolution(self, func, evaluations_left):
        for _ in range(evaluations_left):
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, self.population[i])
                trial_value = func(trial)

                if trial_value < self.best_value:
                    self.best_solution, self.best_value = trial, trial_value

                if trial_value < func(self.population[i]):
                    self.population[i] = trial

                evaluations_left -= 1
                if evaluations_left <= 0:
                    return

    def local_search(self, func, solution):
        step_size = 0.1
        improved = True
        current_value = func(solution)
        while improved:
            improved = False
            for i in range(self.dim):
                for direction in [-1, 1]:
                    candidate = solution.copy()
                    candidate[i] += direction * step_size
                    candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                    candidate_value = func(candidate)
                    if candidate_value < current_value:
                        solution, current_value = candidate, candidate_value
                        improved = True
                        if current_value < self.best_value:
                            self.best_solution, self.best_value = solution, current_value
        return solution

    def __call__(self, func):
        evaluations_left = self.budget
        self.differential_evolution(func, evaluations_left // 2)
        self.best_solution = self.local_search(func, self.best_solution)
        return self.best_solution
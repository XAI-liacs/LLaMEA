import numpy as np

class AdaptiveDEQuantumAdaptiveMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F_min = 0.5
        self.F_max = 1.0
        self.CR = 0.9
        self.mutation_rate = 0.1
        self.potential_solutions = None
        self.best_solution = None
        self.best_score = float('inf')

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.potential_solutions = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def evaluate(self, func):
        scores = np.array([func(sol) for sol in self.potential_solutions])
        for i in range(self.population_size):
            if scores[i] < self.best_score:
                self.best_score = scores[i]
                self.best_solution = self.potential_solutions[i]
        return scores

    def adaptive_mutation(self, iteration, max_iterations):
        scale_factor = self.F_min + (self.F_max - self.F_min) * (1 - (iteration / max_iterations))
        return scale_factor * (np.random.randn(self.dim) * self.mutation_rate)

    def evolve(self, func, iteration, max_iterations):
        new_solutions = np.copy(self.potential_solutions)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = np.random.choice(idxs, 3, replace=False)
            mutant_vector = self.potential_solutions[a] + self.adaptive_mutation(iteration, max_iterations) * (self.potential_solutions[b] - self.potential_solutions[c])
            cross_points = np.random.rand(self.dim) < self.CR
            trial_vector = np.where(cross_points, mutant_vector, self.potential_solutions[i])
            trial_vector = np.clip(trial_vector, func.bounds.lb, func.bounds.ub)
            trial_score = func(trial_vector)
            if trial_score < func(self.potential_solutions[i]):
                new_solutions[i] = trial_vector
        self.potential_solutions = new_solutions

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        max_iterations = self.budget // self.population_size
        iteration = 0
        while func_calls < self.budget:
            self.evaluate(func)
            func_calls += self.population_size
            self.evolve(func, iteration, max_iterations)
            iteration += 1

        return self.best_solution, self.best_score
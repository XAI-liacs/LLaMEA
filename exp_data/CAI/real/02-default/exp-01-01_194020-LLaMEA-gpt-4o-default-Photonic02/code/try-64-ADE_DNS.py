import numpy as np

class ADE_DNS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.mutation_factor = 0.5
        self.crossover_prob = 0.9
        self.best_solution = None
        self.best_value = float('inf')

    def initialize_population(self, lb, ub):
        return [lb + (ub - lb) * np.random.rand(self.dim) for _ in range(self.pop_size)]

    def mutate(self, population, idx, lb, ub):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = population[a] + self.mutation_factor * (population[b] - population[c])
        return np.clip(mutant, lb, ub)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def adaptive_mutation(self, target_value, global_best_value):
        return self.mutation_factor * (target_value / global_best_value)

    def dynamic_neighborhood_search(self, population, idx, lb, ub):
        distances = [np.linalg.norm(population[idx] - population[j]) for j in range(self.pop_size)]
        sorted_indices = np.argsort(distances)
        for j in sorted_indices[:5]:  # Consider the 5 nearest neighbors
            if j != idx:
                trial = self.mutate(population, j, lb, ub)
                trial = self.crossover(population[j], trial)
                trial_value = self.evaluate(trial)
                if trial_value < self.best_value:
                    self.best_value = trial_value
                    self.best_solution = trial

    def evaluate(self, solution):
        return func(solution)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        population = self.initialize_population(lb, ub)
        while evaluations < self.budget:
            for i in range(self.pop_size):
                target = population[i]
                mutant = self.mutate(population, i, lb, ub)
                trial = self.crossover(target, mutant)
                trial_value = func(trial)
                evaluations += 1

                if trial_value < self.evaluate(target):
                    population[i] = trial
                    if trial_value < self.best_value:
                        self.best_value = trial_value
                        self.best_solution = trial

                if evaluations >= self.budget:
                    break

            for i in range(self.pop_size):
                self.dynamic_neighborhood_search(population, i, lb, ub)

        return self.best_solution, self.best_value
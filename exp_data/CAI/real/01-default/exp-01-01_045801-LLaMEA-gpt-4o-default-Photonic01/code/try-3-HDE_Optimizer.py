import numpy as np

class HDE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.F = 0.8  # mutation factor
        self.CR = 0.9  # crossover probability
        self.global_best_position = None
        self.global_best_score = float('inf')

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.scores = np.array([self.evaluate(ind) for ind in self.population])
        best_idx = np.argmin(self.scores)
        self.global_best_position = self.population[best_idx].copy()
        self.global_best_score = self.scores[best_idx]

    def evaluate(self, solution):
        return self.func(solution)

    def mutate(self, idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant_vector = self.population[a] + self.F * (self.population[b] - self.population[c])
        return np.clip(mutant_vector, self.lb, self.ub)

    def crossover(self, target, mutant):
        crossover_vector = np.array([mutant[i] if np.random.rand() < self.CR else target[i] for i in range(self.dim)])
        return crossover_vector

    def local_search(self, candidate):
        perturbation = np.random.normal(0, 0.1, self.dim)
        perturbed_candidate = candidate + perturbation
        return np.clip(perturbed_candidate, self.lb, self.ub)

    def __call__(self, func):
        self.func = func
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        evaluations = 0

        self.initialize_population(self.lb, self.ub)

        while evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial = self.local_search(trial)  # Dynamic local search
                trial_score = self.evaluate(trial)
                evaluations += 1

                if trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial

                if evaluations >= self.budget:
                    break

        return {'solution': self.global_best_position, 'fitness': self.global_best_score}
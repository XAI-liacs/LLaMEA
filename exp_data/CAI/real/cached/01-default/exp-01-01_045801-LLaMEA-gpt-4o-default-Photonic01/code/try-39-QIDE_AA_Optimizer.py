import numpy as np
import heapq

class QIDE_AA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.initial_population_size = self.population_size
        self.F_base = 0.5  # Base mutation factor
        self.CR = 0.9  # Crossover probability
        self.archive_size = 5  # Size of the adaptive archive
        self.population = None
        self.scores = None
        self.archive = []
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.qbits = np.ones((self.population_size, self.dim)) / np.sqrt(2)

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.scores = np.array([self.evaluate(ind) for ind in self.population])
        best_idx = np.argmin(self.scores)
        self.best_solution = self.population[best_idx].copy()
        self.best_score = self.scores[best_idx]
        self.archive = [(self.scores[i], self.population[i].copy()) for i in range(self.archive_size)]
        heapq.heapify(self.archive)

    def evaluate(self, solution):
        return self.func(solution)

    def mutate(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutation_factor = self.F_base * (1 - self.evaluations / self.budget)
        mutant = a + mutation_factor * (b - c)
        return mutant

    def quantum_crossover(self, target, mutant):
        q_mask = np.random.rand(self.dim) < self.qbits[target]
        return np.where(q_mask, mutant, target)

    def select(self, target_idx, trial):
        trial_score = self.evaluate(trial)
        if trial_score < self.scores[target_idx]:
            self.population[target_idx] = trial
            self.scores[target_idx] = trial_score
            self.qbits[target_idx] = np.clip(self.qbits[target_idx] + 0.1 * (trial < self.population[target_idx]), 0, 1)
            if trial_score < self.best_score:
                self.best_score = trial_score
                self.best_solution = trial.copy()
            if len(self.archive) < self.archive_size:
                heapq.heappush(self.archive, (trial_score, trial.copy()))
            else:
                heapq.heappushpop(self.archive, (trial_score, trial.copy()))

    def resize_population(self):
        self.population_size = max(5, int(self.initial_population_size * (1 - self.evaluations / self.budget)))
        self.population = self.population[:self.population_size]
        self.scores = self.scores[:self.population_size]
        self.qbits = self.qbits[:self.population_size]

    def incorporate_archive(self):
        if len(self.archive) == 0:
            return
        archive_indices = np.random.choice(len(self.archive), min(3, len(self.archive)), replace=False)
        for i in archive_indices:
            candidate = self.archive[i][1]
            candidate_mutant = self.mutate_from_archive(candidate)
            candidate = self.quantum_crossover(candidate, candidate_mutant)
            candidate = np.clip(candidate, self.func.bounds.lb, self.func.bounds.ub)
            self.select(np.random.randint(self.population_size), candidate)

    def mutate_from_archive(self, target):
        a, b, c = self.population[np.random.choice(self.population_size, 3, replace=False)]
        mutation_factor = self.F_base * (1 - self.evaluations / self.budget)
        mutant = a + mutation_factor * (b - c)
        return mutant

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            self.resize_population()
            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.quantum_crossover(self.population[i], mutant)
                trial = np.clip(trial, lb, ub)
                self.select(i, trial)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break

            self.incorporate_archive()

        return {'solution': self.best_solution, 'fitness': self.best_score}
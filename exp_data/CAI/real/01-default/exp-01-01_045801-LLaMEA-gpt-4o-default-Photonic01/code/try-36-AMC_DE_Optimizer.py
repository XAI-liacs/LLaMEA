import numpy as np
import heapq

class AMC_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.initial_population_size = self.population_size
        self.F_base = 0.5
        self.CR = 0.9
        self.archive_size = 5
        self.num_cohorts = 3  # Number of cohorts for enhanced exploration
        self.cohorts = [[] for _ in range(self.num_cohorts)]
        self.population = None
        self.scores = None
        self.archive = []
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        full_population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        for i, ind in enumerate(full_population):
            self.cohorts[i % self.num_cohorts].append(ind)
        self.population = np.array(full_population)
        self.scores = np.array([self.evaluate(ind) for ind in self.population])
        best_idx = np.argmin(self.scores)
        self.best_solution = self.population[best_idx].copy()
        self.best_score = self.scores[best_idx]
        self.archive = [(self.scores[i], self.population[i].copy()) for i in range(self.archive_size)]
        heapq.heapify(self.archive)

    def evaluate(self, solution):
        return self.func(solution)

    def mutate(self, target_idx, cohort):
        indices = list(range(len(cohort)))
        indices.remove(target_idx)
        a, b, c = np.array(cohort)[np.random.choice(indices, 3, replace=False)]
        mutation_factor = self.F_base * (1 - self.evaluations / self.budget)
        mutant = a + mutation_factor * (b - c)
        return mutant

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        return np.where(crossover_mask, mutant, target)

    def select(self, target_idx, trial, cohort):
        trial_score = self.evaluate(trial)
        if trial_score < self.scores[target_idx]:
            self.population[target_idx] = trial
            cohort[target_idx % len(cohort)] = trial
            self.scores[target_idx] = trial_score
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

    def incorporate_archive(self):
        if len(self.archive) == 0:
            return
        archive_indices = np.random.choice(len(self.archive), min(3, len(self.archive)), replace=False)
        for i in archive_indices:
            candidate = self.archive[i][1]
            candidate_mutant = self.mutate_from_archive(candidate)
            candidate = self.crossover(candidate, candidate_mutant)
            candidate = np.clip(candidate, self.func.bounds.lb, self.func.bounds.ub)
            self.select(np.random.randint(self.population_size), candidate, self.cohorts[np.random.randint(self.num_cohorts)])

    def mutate_from_archive(self, target):
        a, b, c = self.population[np.random.choice(self.population_size, 3, replace=False)]
        mutation_factor = self.F_base * (1 - self.evaluations / self.budget)
        mutant = a + mutation_factor * (b - c)
        return mutant

    def migrate_between_cohorts(self):
        for i in range(self.num_cohorts):
            if np.random.rand() < 0.2:  # Migration probability
                target_cohort_idx = (i + 1) % self.num_cohorts
                swap_idx = np.random.randint(len(self.cohorts[i]))
                self.cohorts[target_cohort_idx].append(self.cohorts[i].pop(swap_idx))

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            self.resize_population()
            for cohort in self.cohorts:
                for i in range(len(cohort)):
                    mutant = self.mutate(i, cohort)
                    trial = self.crossover(cohort[i], mutant)
                    trial = np.clip(trial, lb, ub)
                    self.select(i, trial, cohort)
                    self.evaluations += 1
                    if self.evaluations >= self.budget:
                        break

            self.incorporate_archive()
            self.migrate_between_cohorts()

        return {'solution': self.best_solution, 'fitness': self.best_score}
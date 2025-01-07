import numpy as np

class QE_ADE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 30
        self.population_size = self.initial_population_size
        self.F_min = 0.4
        self.F_max = 0.9
        self.CR_min = 0.1
        self.CR_max = 0.9
        self.population = None
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.scores = np.array([self.evaluate(ind) for ind in self.population])
        self.update_best()

    def evaluate(self, solution):
        return self.func(solution)

    def mutate(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        F = np.random.uniform(self.F_min, self.F_max)
        mutant = a + F * (b - c)
        return mutant

    def crossover(self, target, mutant):
        CR = np.random.uniform(self.CR_min, self.CR_max)
        crossover_mask = np.random.rand(self.dim) < CR
        return np.where(crossover_mask, mutant, target)

    def select(self, target_idx, trial):
        trial_score = self.evaluate(trial)
        if trial_score < self.scores[target_idx]:
            self.population[target_idx] = trial
            self.scores[target_idx] = trial_score
            if trial_score < self.best_score:
                self.best_score = trial_score
                self.best_solution = trial.copy()

    def quantum_perturbation(self):
        for i in range(self.population_size):
            quantum_shift = np.random.uniform(-0.1, 0.1, self.dim)
            perturbed_solution = self.population[i] + quantum_shift
            perturbed_solution = np.clip(perturbed_solution, self.func.bounds.lb, self.func.bounds.ub)
            self.select(i, perturbed_solution)

    def update_best(self):
        best_idx = np.argmin(self.scores)
        self.best_solution = self.population[best_idx].copy()
        self.best_score = self.scores[best_idx]

    def dynamic_population_resizing(self):
        if self.evaluations > self.budget / 2 and self.population_size > self.initial_population_size / 2:
            self.population_size = max(self.initial_population_size // 2, int(self.population_size * 0.9))
            self.population = self.population[:self.population_size]
            self.scores = self.scores[:self.population_size]

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial = np.clip(trial, lb, ub)
                self.select(i, trial)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break
            self.quantum_perturbation()
            self.dynamic_population_resizing()

        return {'solution': self.best_solution, 'fitness': self.best_score}
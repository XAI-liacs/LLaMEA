import numpy as np

class QS_ADE_LS_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.F_min = 0.4
        self.F_max = 0.9
        self.CR_min = 0.1
        self.CR_max = 0.9
        self.population = None
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.successful_F = []
        self.successful_CR = []

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
        F = self.adapt_parameter(self.F_min, self.F_max, self.successful_F)
        mutant = a + F * (b - c)
        return mutant

    def crossover(self, target, mutant):
        CR = self.adapt_parameter(self.CR_min, self.CR_max, self.successful_CR)
        crossover_mask = np.random.rand(self.dim) < CR
        return np.where(crossover_mask, mutant, target)

    def select(self, target_idx, trial, F, CR):
        trial_score = self.evaluate(trial)
        if trial_score < self.scores[target_idx]:
            self.population[target_idx] = trial
            self.scores[target_idx] = trial_score
            self.successful_F.append(F)
            self.successful_CR.append(CR)
            if trial_score < self.best_score:
                self.best_score = trial_score
                self.best_solution = trial.copy()

    def adapt_parameter(self, min_val, max_val, successes):
        if successes:
            return np.clip(np.mean(successes) + np.random.normal(scale=0.1), min_val, max_val)
        else:
            return np.random.uniform(min_val, max_val)

    def quantum_inspired_mutation(self):
        for i in range(self.population_size):
            quantum_superposition = np.random.uniform(-1, 1, self.dim)
            quantum_mutant = self.best_solution + quantum_superposition * np.abs(self.population[i] - self.best_solution)
            quantum_mutant = np.clip(quantum_mutant, self.func.bounds.lb, self.func.bounds.ub)
            self.select(i, quantum_mutant, None, None)

    def update_best(self):
        best_idx = np.argmin(self.scores)
        self.best_solution = self.population[best_idx].copy()
        self.best_score = self.scores[best_idx]

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial = np.clip(trial, lb, ub)
                F = self.adapt_parameter(self.F_min, self.F_max, self.successful_F)
                CR = self.adapt_parameter(self.CR_min, self.CR_max, self.successful_CR)
                self.select(i, trial, F, CR)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break
            self.quantum_inspired_mutation()
            self.evaluations += self.population_size

        return {'solution': self.best_solution, 'fitness': self.best_score}
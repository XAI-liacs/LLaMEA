import numpy as np

class CGIO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
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

    def generate_wave_interference(self, a, b, c):
        wave1 = np.sin(a)  # Coherent wave from solution a
        wave2 = np.cos(b)  # Coherent wave from solution b
        interference = wave1 + wave2
        interference_mutant = c + interference * (b - a)
        return interference_mutant

    def mutate(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant = self.generate_wave_interference(a, b, c)
        return mutant

    def crossover(self, target, mutant):
        CR = 0.8  # Fixed crossover rate
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

    def coherence_guided_mutation(self):
        for i in range(self.population_size):
            indices = np.random.choice(range(self.population_size), 2, replace=False)
            a, b = self.population[indices]
            coherence_mutant = self.best_solution + np.sin(a - b) * (self.best_solution - self.population[i])
            coherence_mutant = np.clip(coherence_mutant, self.func.bounds.lb, self.func.bounds.ub)
            self.select(i, coherence_mutant)

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
                self.select(i, trial)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break
            self.coherence_guided_mutation()
            self.evaluations += self.population_size

        return {'solution': self.best_solution, 'fitness': self.best_score}
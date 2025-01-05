import numpy as np

class QADE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.initial_population_size = self.population_size
        self.F_base = 0.5  # Base mutation factor
        self.CR_base = 0.7  # Base crossover probability
        self.population = None
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.scores = np.array([self.evaluate(ind) for ind in self.population])
        best_idx = np.argmin(self.scores)
        self.best_solution = self.population[best_idx].copy()
        self.best_score = self.scores[best_idx]

    def evaluate(self, solution):
        return self.func(solution)

    def mutate(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutation_factor = self.F_base * (1 - self.evaluations / self.budget)
        mutant = a + mutation_factor * (b - c)
        return mutant

    def quantum_rotation(self, vec):
        theta = np.pi * np.random.uniform(0, 1, len(vec))
        q_rot = np.cos(theta) * vec + np.sin(theta) * np.random.uniform(-1, 1, len(vec))
        return q_rot

    def adaptive_crossover(self, target, mutant):
        # Adapt crossover probability based on current evaluation ratio
        CR = self.CR_base + 0.2 * (self.evaluations / self.budget)
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

    def resize_population(self):
        self.population_size = max(5, int(self.initial_population_size * (1 - self.evaluations / self.budget)))
        self.population = self.population[:self.population_size]
        self.scores = self.scores[:self.population_size]

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            self.resize_population()
            for i in range(self.population_size):
                mutant = self.mutate(i)
                quantum_mutant = self.quantum_rotation(mutant)
                trial = self.adaptive_crossover(self.population[i], quantum_mutant)
                trial = np.clip(trial, lb, ub)
                self.select(i, trial)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break

        return {'solution': self.best_solution, 'fitness': self.best_score}
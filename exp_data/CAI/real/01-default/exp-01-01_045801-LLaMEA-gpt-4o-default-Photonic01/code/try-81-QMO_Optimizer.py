import numpy as np

class QMO_Optimizer:
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

    def quantum_morphogenetic_mutation(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b = self.population[np.random.choice(indices, 2, replace=False)]
        alpha = np.random.uniform(0, 1)
        beta = np.random.uniform(0, 1)
        quantum_factor = np.random.uniform(-1, 1, self.dim)

        morphogenetic_mutant = alpha * a + (1 - alpha) * b + beta * quantum_factor * (self.best_solution - self.population[target_idx])
        morphogenetic_mutant = np.clip(morphogenetic_mutant, self.func.bounds.lb, self.func.bounds.ub)
        
        return morphogenetic_mutant

    def select(self, target_idx, trial):
        trial_score = self.evaluate(trial)
        if trial_score < self.scores[target_idx]:
            self.population[target_idx] = trial
            self.scores[target_idx] = trial_score
            if trial_score < self.best_score:
                self.best_score = trial_score
                self.best_solution = trial.copy()

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
                trial = self.quantum_morphogenetic_mutation(i)
                self.select(i, trial)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break
        
        return {'solution': self.best_solution, 'fitness': self.best_score}
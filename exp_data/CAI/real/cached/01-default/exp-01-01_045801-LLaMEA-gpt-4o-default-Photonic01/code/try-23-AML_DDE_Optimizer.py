import numpy as np

class AML_DDE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.initial_population_size = self.population_size
        self.F_base = 0.5  # Base mutation factor
        self.CR = 0.9  # Crossover probability
        self.population = None
        self.scores = None
        self.best_solutions = []
        self.best_scores = []
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.scores = np.array([self.evaluate(ind) for ind in self.population])
        self.update_best_solutions()

    def evaluate(self, solution):
        return self.func(solution)

    def mutate(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        # Select multiple leaders from the best solutions
        leaders = np.random.choice(len(self.best_solutions), 3, replace=False)
        a, b, c = [self.best_solutions[i] for i in leaders]
        # Adapt mutation factor based on diversity measure
        diversity = np.std(self.scores)
        mutation_factor = self.F_base * (1 - diversity / np.max(self.scores))
        mutant = a + mutation_factor * (b - c)
        return mutant

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        return np.where(crossover_mask, mutant, target)

    def select(self, target_idx, trial):
        trial_score = self.evaluate(trial)
        if trial_score < self.scores[target_idx]:
            self.population[target_idx] = trial
            self.scores[target_idx] = trial_score
            if trial_score < np.min(self.best_scores):
                self.best_solutions.append(trial.copy())
                self.best_scores.append(trial_score)
                self.prune_best_solutions()

    def prune_best_solutions(self):
        # Keep only the top few best solutions
        if len(self.best_scores) > 5:
            worst_idx = np.argmax(self.best_scores)
            del self.best_scores[worst_idx]
            del self.best_solutions[worst_idx]

    def update_best_solutions(self):
        best_idx = np.argmin(self.scores)
        if not self.best_scores or self.scores[best_idx] < np.min(self.best_scores):
            self.best_solutions = [self.population[best_idx].copy()]
            self.best_scores = [self.scores[best_idx]]

    def resize_population(self):
        # Dynamically resize the population
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
                trial = self.crossover(self.population[i], mutant)
                trial = np.clip(trial, lb, ub)
                self.select(i, trial)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break

        best_idx = np.argmin(self.best_scores)
        best_solution = self.best_solutions[best_idx]
        best_score = self.best_scores[best_idx]
        return {'solution': best_solution, 'fitness': best_score}
import numpy as np

class AMDE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.local_search_prob = 0.1  # Probability of applying local search
        self.F_base = 0.5  # Base mutation factor
        self.CR = 0.9  # Crossover probability
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

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        return np.where(crossover_mask, mutant, target)
    
    def local_search(self, solution, lb, ub):
        step_size = 0.1 * (ub - lb)
        perturbation = np.random.uniform(-step_size, step_size, solution.shape)
        new_solution = solution + perturbation
        new_solution = np.clip(new_solution, lb, ub)
        return new_solution

    def select(self, target_idx, trial):
        trial_score = self.evaluate(trial)
        if trial_score < self.scores[target_idx]:
            self.population[target_idx] = trial
            self.scores[target_idx] = trial_score
            if trial_score < self.best_score:
                self.best_score = trial_score
                self.best_solution = trial.copy()

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial = np.clip(trial, lb, ub)

                if np.random.rand() < self.local_search_prob:
                    trial = self.local_search(trial, lb, ub)

                self.select(i, trial)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break

        return {'solution': self.best_solution, 'fitness': self.best_score}
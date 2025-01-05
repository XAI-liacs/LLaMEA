import numpy as np

class BI_MAS_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.evaporation_rate = 0.1
        self.pheromone_matrix = None
        self.population = None
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.scores = np.array([self.evaluate(ind) for ind in self.population])
        self.update_best()
        self.pheromone_matrix = np.full((self.population_size, self.dim), 1.0)

    def evaluate(self, solution):
        return self.func(solution)

    def update_pheromone(self, idx, increment):
        self.pheromone_matrix[idx] *= (1 - self.evaporation_rate)
        self.pheromone_matrix[idx] += increment

    def mutate(self, idx):
        neighbors = np.random.choice(range(self.population_size), 2, replace=False)
        a, b = self.population[neighbors]
        mutant = self.population[idx] + np.random.rand() * (a - b)
        return mutant

    def crossover(self, target, mutant):
        pheromones = self.pheromone_matrix[target]
        prob = pheromones / pheromones.sum()
        crossover_mask = np.random.rand(self.dim) < prob
        return np.where(crossover_mask, mutant, target)

    def select(self, target_idx, trial):
        trial_score = self.evaluate(trial)
        if trial_score < self.scores[target_idx]:
            self.population[target_idx] = trial
            self.scores[target_idx] = trial_score
            if trial_score < self.best_score:
                self.best_score = trial_score
                self.best_solution = trial.copy()
            self.update_pheromone(target_idx, 1.0 / (1.0 + trial_score))

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

        return {'solution': self.best_solution, 'fitness': self.best_score}
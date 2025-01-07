import numpy as np

class ODE_AC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget)
        self.population = None
        self.scores = None
        self.best_solution = None
        self.best_score = np.inf
        self.bounds = None
        self.mutation_factor = 0.5  # Initial mutation factor
        self.crossover_rate = 0.7  # Crossover rate
        self.oscillation_factor = 0.05  # Controls mutation factor oscillation
        self.adapt_rate = 0.3  # Rate at which adaptive control mechanisms are applied

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.scores = np.full(self.population_size, np.inf)
        self.bounds = (lb, ub)

    def mutate(self, target_idx):
        idxs = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant = a + self.mutation_factor * (b - c)
        lb, ub = self.bounds
        return np.clip(mutant, lb, ub)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def adaptive_oscillation(self, generation):
        self.mutation_factor = 0.5 + self.oscillation_factor * np.sin(2 * np.pi * generation / self.budget)

    def adaptive_mutation_control(self):
        diversity = np.std(self.population, axis=0).mean()
        max_diversity = np.sqrt(np.sum((self.bounds[1] - self.bounds[0])**2))
        self.mutation_factor = 0.4 + 0.3 * (diversity / max_diversity)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0
        generation = 0

        while evaluations < self.budget:
            generation += 1
            self.adaptive_oscillation(generation)

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)

                trial_score = func(trial)
                evaluations += 1

                if trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score

                if trial_score < self.best_score:
                    self.best_score = trial_score
                    self.best_solution = trial

            if np.random.rand() < self.adapt_rate:
                self.adaptive_mutation_control()

        return self.best_solution, self.best_score
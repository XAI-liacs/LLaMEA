import numpy as np

class AdaptiveHybridDifferentialEvolutionLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.init_population_size = 10 * dim
        self.F = 0.8  # Initial DE scaling factor
        self.CR = 0.9  # Crossover probability
        self.local_search_rate = 0.1  # Probability of applying local search
        self.population = None
        self.best_individual = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.adaptation_rate = 0.1  # Rate to adapt mutation factor
        self.dynamic_population = True

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        self.initialize_population(bounds, func)
        
        while self.evaluations < self.budget:
            if self.dynamic_population and self.evaluations % (self.budget // 10) == 0:
                self.reduce_population()

            for i in range(len(self.population)):
                if self.evaluations >= self.budget:
                    break
                
                # Differential Evolution mutation and crossover
                mutant = self.de_mutation(i, bounds)
                trial = self.de_crossover(self.population[i], mutant)
                
                # Local search
                if np.random.rand() < self.local_search_rate:
                    trial = self.local_search(trial, bounds, func)
                
                # Selection
                trial_score = self.evaluate(func, trial)
                if trial_score < self.evaluate(func, self.population[i]):
                    self.population[i] = trial
                    if trial_score < self.best_score:
                        self.best_score = trial_score
                        self.best_individual = trial
                        self.F = min(1.0, self.F + self.adaptation_rate)  # Increase mutation factor adaptively
                else:
                    self.F = max(0.4, self.F - self.adaptation_rate)  # Decrease mutation factor

        return self.best_individual

    def initialize_population(self, bounds, func):
        self.population = np.random.rand(self.init_population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        for individual in self.population:
            score = self.evaluate(func, individual)
            if score < self.best_score:
                self.best_score = score
                self.best_individual = individual

    def reduce_population(self):
        if len(self.population) > 4 * self.dim:
            self.population = self.population[:len(self.population)//2]

    def de_mutation(self, target_idx, bounds):
        indices = list(range(len(self.population)))
        indices.remove(target_idx)
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), bounds[:, 0], bounds[:, 1])
        return mutant

    def de_crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def local_search(self, individual, bounds, func):
        perturbation = np.random.normal(0, 0.02, self.dim)
        perturbed_individual = np.clip(individual + perturbation, bounds[:, 0], bounds[:, 1])
        perturbed_score = self.evaluate(func, perturbed_individual)
        if perturbed_score < self.evaluate(func, individual):
            return perturbed_individual
        return individual

    def evaluate(self, func, individual):
        if self.evaluations >= self.budget:
            return float('inf')
        self.evaluations += 1
        return func(individual)
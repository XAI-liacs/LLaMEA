import numpy as np

class HybridDifferentialEvolutionLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.8  # DE scaling factor
        self.CR = 0.9  # Crossover probability
        self.local_search_rate = 0.1  # Probability of applying local search
        self.population = None
        self.best_individual = None
        self.best_score = float('inf')
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        self.initialize_population(bounds, func)  # Pass func to initialize_population
        
        while self.evaluations < self.budget:
            for i in range(self.population_size):
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
        
        return self.best_individual

    def initialize_population(self, bounds, func):  # Add func as parameter
        self.population = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        for individual in self.population:
            score = self.evaluate(func, individual)
            if score < self.best_score:
                self.best_score = score
                self.best_individual = individual

    def de_mutation(self, target_idx, bounds):
        indices = list(range(self.population_size))
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
        perturbation = np.random.normal(0, 0.05, self.dim)  # Reduced perturbation scale
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
import numpy as np

class LevyFlightEnhancedDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha = 0.5
        self.beta = 1.5
        self.scaling_factor_min = 0.5
        self.scaling_factor_max = 1.0
        self.crossover_probability = 0.9
        self.population = None
        self.best_individual = None
        self.best_score = float('inf')

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.best_individual = np.copy(self.population[0])

    def evaluate(self, func):
        scores = np.array([func(ind) for ind in self.population])
        best_idx = np.argmin(scores)
        if scores[best_idx] < self.best_score:
            self.best_score = scores[best_idx]
            self.best_individual = np.copy(self.population[best_idx])
        return scores

    def levy_flight(self):
        u = np.random.normal(0, 1, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v)**(1/self.beta)
        return self.alpha * step

    def mutate(self, target_idx, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        idxs = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = np.clip(self.population[a] + self.dynamic_scaling_factor() * (self.population[b] - self.population[c]), lb, ub)
        return mutant

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_probability
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def dynamic_scaling_factor(self):
        return self.scaling_factor_min + (self.scaling_factor_max - self.scaling_factor_min) * np.random.rand()

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            for i in range(self.population_size):
                mutant = self.mutate(i, func.bounds)
                trial = self.crossover(self.population[i], mutant)
                
                # Perform Lévy flight
                trial += self.levy_flight()
                
                trial_score = func(trial)
                func_calls += 1
                if trial_score < scores[i]:
                    self.population[i] = trial
                    scores[i] = trial_score
                if trial_score < self.best_score:
                    self.best_score = trial_score
                    self.best_individual = trial
                if func_calls >= self.budget:
                    break

        return self.best_individual, self.best_score
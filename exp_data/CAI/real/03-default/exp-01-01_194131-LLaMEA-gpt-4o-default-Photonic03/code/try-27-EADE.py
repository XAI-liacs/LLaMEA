import numpy as np

class EADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.population = np.random.rand(self.population_size, dim)
        self.population_scores = np.full(self.population_size, float('inf'))
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.F = 0.5  # mutation factor
        self.CR = 0.9  # crossover probability
        self.beta = 0.5

    def levy_flight(self, scale=0.01):
        u = np.random.normal(0, 1, self.dim) * scale
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / 3))
        return step

    def differential_evolution(self, idx, func):
        candidates = [i for i in range(self.population_size) if i != idx]
        a, b, c = self.population[np.random.choice(candidates, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
        trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.population[idx])
        
        # Quantum-inspired mutation with Levy flight
        if np.random.rand() < self.beta:
            trial += self.levy_flight()

        trial = np.clip(trial, func.bounds.lb, func.bounds.ub)
        trial_score = func(trial)
        if trial_score < self.population_scores[idx]:
            self.population[idx] = trial
            self.population_scores[idx] = trial_score
            if trial_score < self.best_score:
                self.best_solution = trial
                self.best_score = trial_score
        self.evaluations += 1

    def adapt_population(self):
        if self.evaluations % (self.budget // 4) == 0:
            new_population_size = min(self.population_size + 5, 20 * self.dim)
            if new_population_size > self.population_size:
                additional_population = np.random.rand(new_population_size - self.population_size, self.dim)
                self.population = np.vstack((self.population, additional_population))
                self.population_scores = np.hstack((self.population_scores, np.full(new_population_size - self.population_size, float('inf'))))
                self.population_size = new_population_size

    def __call__(self, func):
        self.population = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            score = func(self.population[i])
            self.population_scores[i] = score
            if score < self.best_score:
                self.best_solution = self.population[i]
                self.best_score = score
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.best_solution
        
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                self.differential_evolution(i, func)
                if self.evaluations >= self.budget:
                    break
            self.adapt_population()

        return self.best_solution
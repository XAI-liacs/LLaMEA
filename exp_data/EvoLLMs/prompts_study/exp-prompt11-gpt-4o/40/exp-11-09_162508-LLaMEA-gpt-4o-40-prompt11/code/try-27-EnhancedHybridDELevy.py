import numpy as np

class EnhancedHybridDELevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Increased population size for diversity
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.7  # Adjusted mutation factor for improved control
        self.CR = 0.8  # Adjusted crossover probability for better diversity
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.scores = np.full(self.population_size, np.inf)
        self.func_evals = 0

    def levy_flight(self, L):
        beta = 1.5
        sigma_u = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / (abs(v) ** (1 / beta))
        return step

    def __call__(self, func):
        best = None
        best_score = np.inf
        
        def evaluate(ind):
            nonlocal best, best_score
            if self.func_evals < self.budget:
                score = func(ind)
                self.func_evals += 1
                if score < best_score:
                    best_score = score
                    best = ind
                return score
            else:
                return None

        for i in range(self.population_size):
            self.scores[i] = evaluate(self.population[i])

        while self.func_evals < self.budget:
            ranked_indices = np.argsort(self.scores)
            for i in range(self.population_size):
                # Rank-based selection for diversity
                if i < self.population_size // 2:
                    # Select from top half
                    a, b, c = np.random.choice(ranked_indices[:self.population_size // 2], 3, replace=False)
                else:
                    # Select from entire population
                    indices = list(range(self.population_size))
                    indices.remove(i)
                    a, b, c = np.random.choice(indices, 3, replace=False)

                # Mutation
                mutant = self.population[a] + self.F * (self.population[b] - self.population[c])

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Possible Lévy flight step
                levy_step = self.levy_flight(1.5) * (trial - best)
                trial = trial + levy_step
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                # Selection
                trial_score = evaluate(trial)
                if trial_score is not None and trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score

        return best, best_score
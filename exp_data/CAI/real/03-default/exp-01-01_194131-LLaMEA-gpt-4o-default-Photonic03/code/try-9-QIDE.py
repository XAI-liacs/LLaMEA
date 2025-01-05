import numpy as np

class QIDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.scores = np.full(self.population_size, float('inf'))
        self.best_individual = None
        self.best_score = float('inf')
        self.crossover_rate = 0.9
        self.f = 0.5  # Differential weight
        self.evaluations = 0

    def stochastic_tunneling(self, score):
        # Apply stochastic tunneling transformation
        if score < self.best_score:
            return score
        return score * np.exp(-self.f * (score - self.best_score)**2)

    def differential_evolution(self, idx, func):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        
        mutant = self.positions[a] + self.f * (self.positions[b] - self.positions[c])
        trial = np.copy(self.positions[idx])

        jrand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() < self.crossover_rate or j == jrand:
                trial[j] = mutant[j]
        
        trial = np.clip(trial, func.bounds.lb, func.bounds.ub)
        trial_score = func(trial)
        trial_score = self.stochastic_tunneling(trial_score)

        if trial_score < self.scores[idx]:
            self.positions[idx] = trial
            self.scores[idx] = trial_score

        if trial_score < self.best_score:
            self.best_individual = trial
            self.best_score = trial_score

        self.evaluations += 1

    def adapt_crossover_rate(self):
        if self.evaluations % (self.budget // 10) == 0:
            self.crossover_rate = np.clip(self.crossover_rate + np.random.uniform(-0.1, 0.1), 0.1, 1.0)

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            score = func(self.positions[i])
            self.scores[i] = self.stochastic_tunneling(score)
            if score < self.best_score:
                self.best_individual = self.positions[i]
                self.best_score = score
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.best_individual

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                self.differential_evolution(i, func)
                if self.evaluations >= self.budget:
                    break
            self.adapt_crossover_rate()

        return self.best_individual
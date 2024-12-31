import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)

    def __call__(self, func):
        evals = 0
        self.fitness = np.array([func(ind) for ind in self.population])
        evals += self.pop_size

        while evals < self.budget:
            trial_population = np.copy(self.population)
            improvements = 0
            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
                trial = np.copy(self.population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        trial[j] = mutant[j]
                trial = np.clip(trial, -5.0, 5.0)
                f_trial = func(trial)
                evals += 1
                if f_trial < self.fitness[i]:
                    trial_population[i] = trial
                    self.fitness[i] = f_trial
                    improvements += 1
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                if evals >= self.budget:
                    break
            self.population = trial_population
            improvement_rate = improvements / self.pop_size
            if improvement_rate > 0.1:  # Dynamically adjust F based on improvement rate
                self.F = min(1.0, self.F + 0.1)
            else:
                self.F = max(0.5, self.F - 0.1)
            if np.std(self.fitness) > 0.01:  # Adjust CR based on diversity
                self.CR = 0.9
            else:
                self.CR = 0.7

        return self.f_opt, self.x_opt
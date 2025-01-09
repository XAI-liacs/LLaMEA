import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.NP = 10  # population size
        self.CR = 0.9  # crossover rate
        self.F = 0.8  # differential weight
        self.F_l = 0.5  # lower bound of F
        self.F_u = 1.0  # upper bound of F
        self.P = np.random.uniform(-5.0, 5.0, (self.NP, dim))
    
    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.NP):
                idxs = [idx for idx in range(self.NP) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                F_i = np.random.uniform(self.F_l, self.F_u)
                mutant = self.P[a] + F_i * (self.P[b] - self.P[c])
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, self.P[i])
                if func(trial) < func(self.P[i]):
                    self.P[i] = trial
                evaluations += 1
                if evaluations >= self.budget:
                    break
        best_solution = self.P[np.argmin([func(ind) for ind in self.P])]
        return best_solution
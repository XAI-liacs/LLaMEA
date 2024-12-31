import numpy as np

class DifferentialEvolutionWithLevyFlights:
    def __init__(self, budget=10000, dim=10, pop_size=20, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None
        self.iterations = budget // pop_size
        self.lb = -5.0
        self.ub = 5.0

    def levy_flight(self, lam=1.5):
        sigma = (np.gamma(1 + lam) * np.sin(np.pi * lam / 2) /
                 (np.gamma((1 + lam) / 2) * lam * 2 ** ((lam - 1) / 2))) ** (1 / lam)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v) ** (1 / lam)
        return step

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for gen in range(self.iterations):
            F_adaptive = self.F * (1 - gen / self.iterations)  # Adaptive F
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(population[a] + F_adaptive * (population[b] - population[c]), self.lb, self.ub)
                
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[i])
                
                if np.random.rand() < 0.5:
                    trial += self.levy_flight()
                    trial = np.clip(trial, self.lb, self.ub)
                    
                f_trial = func(trial)
                
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

        return self.f_opt, self.x_opt
import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=20):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = func.bounds
        pop = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = pop[best_idx]

        F = 0.5  # initial mutation factor
        CR = 0.9  # initial crossover probability

        evals = self.population_size

        while evals < self.budget:
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])
                
                f_trial = func(trial)
                evals += 1

                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                if evals >= self.budget:
                    break
            
            F = 0.5 + np.random.rand() * 0.3  # adapt F within [0.5, 0.8]
            CR = 0.8 + np.random.rand() * 0.2  # adapt CR within [0.8, 1.0]

        return self.f_opt, self.x_opt
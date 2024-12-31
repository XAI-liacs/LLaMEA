import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = pop[best_idx]
        
        F = 0.5
        CR = 0.9
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                indices = np.arange(self.population_size)
                np.random.shuffle(indices)
                x1, x2, x3 = pop[indices[:3]]

                mutant = np.clip(x1 + F * (x2 - x3), lb, ub)
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, pop[i])

                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                # Adapt F and CR based on successful trials
                F = 0.5 + 0.3 * np.random.rand()
                CR = 0.9 * np.random.rand()

        return self.f_opt, self.x_opt
import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=30, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Mutation factor
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = np.stack([func.bounds.lb, func.bounds.ub], axis=1)  # Shape (dim, 2)
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.pop_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        while evals < self.budget:
            for i in range(self.pop_size):
                adaptive_F = self.F * (1 - evals / self.budget)
                
                indices = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + adaptive_F * (b - c), func.bounds.lb, func.bounds.ub)

                trial = np.copy(population[i])
                crossover_points = np.random.rand(self.dim) < self.CR
                trial[crossover_points] = mutant[crossover_points]

                f_trial = func(trial)
                evals += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                if evals >= self.budget:
                    break

        return self.f_opt, self.x_opt
import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=20, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = F  # Mutation factor
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = pop[best_idx].copy()
        evals = self.population_size
        
        while evals < self.budget:
            for i in range(self.population_size):
                if evals >= self.budget:
                    break
                
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = pop[indices]
                mutant = np.clip(a + self.F * (b - c), lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, pop[i])

                # Selection
                f_trial = func(trial)
                evals += 1
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial

                    # Update best solution found
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()

        return self.f_opt, self.x_opt
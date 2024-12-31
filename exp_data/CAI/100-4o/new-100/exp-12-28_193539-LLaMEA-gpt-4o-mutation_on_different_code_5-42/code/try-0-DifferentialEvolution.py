import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=15, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        # Find the initial best
        idx_best = np.argmin(fitness)
        self.f_opt = fitness[idx_best]
        self.x_opt = population[idx_best].copy()
        
        eval_count = self.pop_size

        while eval_count < self.budget:
            for i in range(self.pop_size):
                # Select three distinct random indices
                indices = np.arange(self.pop_size)
                indices = np.delete(indices, i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                # Mutate
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                
                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])
                
                # Evaluate trial
                f_trial = func(trial)
                eval_count += 1
                
                # Select
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                
                # Check for new best
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial.copy()

                if eval_count >= self.budget:
                    break

        return self.f_opt, self.x_opt
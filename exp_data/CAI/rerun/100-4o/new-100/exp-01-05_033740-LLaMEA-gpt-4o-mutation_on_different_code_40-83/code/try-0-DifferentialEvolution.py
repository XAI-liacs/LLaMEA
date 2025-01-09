import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size

        # Track the best solution
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        # Optimization loop
        while eval_count < self.budget:
            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + self.F * (x2 - x3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                # Crossover
                trial = np.copy(population[i])
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial[crossover_mask] = mutant[crossover_mask]

                # Selection
                f_trial = func(trial)
                eval_count += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                if eval_count >= self.budget:
                    break

        return self.f_opt, self.x_opt
import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]

        # Budget for function evaluations
        evals = self.pop_size

        # DE parameters
        F_base = 0.5
        CR_base = 0.9

        # Start evolution
        while evals < self.budget:
            # Calculate diversity and adjust F and CR
            diversity = np.std(population, axis=0).mean() / (func.bounds.ub - func.bounds.lb)
            fitness_diversity = np.std(fitness) / (np.max(fitness) - np.min(fitness) + 1e-10)
            F = F_base + 0.1 * (1 - diversity) * fitness_diversity
            CR = CR_base - 0.1 * diversity

            new_population = np.zeros_like(population)
            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + F * (x2 - x3), func.bounds.lb, func.bounds.ub)

                # Crossover
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[i])

                # Evaluate trial individual
                f_trial = func(trial)
                evals += 1

                # Selection
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    new_population[i] = population[i]

                if evals >= self.budget:
                    break
            
            population = new_population

        return self.f_opt, self.x_opt
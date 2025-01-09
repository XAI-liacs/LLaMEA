import numpy as np

class HybridDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size

        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Mutation: select three random individuals a, b, c different from i
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, bounds[0], bounds[1])

                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                if not np.any(crossover):
                    crossover[np.random.randint(self.dim)] = True
                trial = np.where(crossover, mutant, population[i])

                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial

                    # Update best solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                if evaluations >= self.budget:
                    break
        
        # Local search using hill climbing from the best solution found
        step_size = 0.1
        while evaluations < self.budget:
            candidate = self.x_opt + np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(candidate, bounds[0], bounds[1])
            f_candidate = func(candidate)
            evaluations += 1
            if f_candidate < self.f_opt:
                self.f_opt = f_candidate
                self.x_opt = candidate

        return self.f_opt, self.x_opt
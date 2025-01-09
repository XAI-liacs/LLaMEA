import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evals = self.pop_size

        while evals < self.budget:
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                f_trial = func(trial)
                evals += 1

                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
            
            # Adjust population if diversity is low
            if np.std(pop) < 0.1:
                self.F = np.clip(self.F + 0.1 * np.random.randn(), 0.4, 0.9)
                self.CR = np.clip(self.CR + 0.1 * np.random.randn(), 0.7, 1.0)

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]
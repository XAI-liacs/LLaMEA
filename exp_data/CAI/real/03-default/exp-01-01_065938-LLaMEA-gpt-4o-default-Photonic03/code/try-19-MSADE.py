import numpy as np

class MSADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = 0.5   # Scaling factor
        self.CR = 0.9  # Crossover rate
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        best_idx = np.argmin(fitness)
        best_global = pop[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            next_pop = np.zeros_like(pop)

            for i in range(self.population_size):
                indices = np.random.choice(range(self.population_size), 3, replace=False)
                x0, x1, x2 = pop[indices]
                
                if np.random.rand() < 0.5:
                    # DE/rand/1 strategy
                    mutant = x0 + self.F * (x1 - x2)
                else:
                    # DE/best/1 strategy
                    mutant = best_global + self.F * (x1 - x2)

                mutant = np.clip(mutant, lb, ub)

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    next_pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best_global = trial
                else:
                    next_pop[i] = pop[i]

            pop = next_pop
            self.history.append(best_global)

        return best_global
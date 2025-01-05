import numpy as np

class EMSADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.F_min, self.F_max = 0.4, 0.9  # Dynamic scaling factor range
        self.CR = 0.9  # Crossover rate
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.initial_population_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        best_idx = np.argmin(fitness)
        best_global = pop[best_idx]

        evaluations = self.initial_population_size
        population_size = self.initial_population_size

        while evaluations < self.budget:
            next_pop = np.zeros_like(pop)
            F = self.F_min + (self.F_max - self.F_min) * (1 - evaluations / self.budget)

            for i in range(population_size):
                indices = np.random.choice(range(population_size), 3, replace=False)
                x0, x1, x2 = pop[indices]
                
                if np.random.rand() < 0.5:
                    # DE/rand/1 strategy
                    mutant = x0 + F * (x1 - x2)
                else:
                    # DE/best/1 strategy
                    mutant = best_global + F * (x1 - x2)

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

            # Update population size adaptively
            population_size = int(self.initial_population_size * (1 - evaluations / self.budget))
            population_size = max(10, population_size)  # Ensure a minimum population size

            pop = next_pop[:population_size]
            fitness = fitness[:population_size]
            self.history.append(best_global)

        return best_global
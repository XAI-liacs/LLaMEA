import numpy as np

class EMSADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.F = 0.5   # Initial scaling factor
        self.CR = 0.9  # Initial crossover rate
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
            
            # Dynamically adjust population size
            if evaluations % (self.budget // 5) == 0:
                population_size = max(5, population_size // 2)
                pop = pop[:population_size]
                fitness = fitness[:population_size]
                next_pop = next_pop[:population_size]

            for i in range(population_size):
                indices = np.random.choice(range(population_size), 3, replace=False)
                x0, x1, x2 = pop[indices]

                # Adaptive mutation strategy
                if np.random.rand() < 0.5:
                    mutant = x0 + self.F * (x1 - x2)
                else:
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
                    
            # Dynamic adaptive control of parameters
            success_rate = np.count_nonzero(fitness < np.min(fitness)) / float(population_size)
            self.F = 0.5 + 0.3 * success_rate
            self.CR = 0.9 - 0.3 * success_rate

            pop = next_pop
            self.history.append(best_global)

        return best_global
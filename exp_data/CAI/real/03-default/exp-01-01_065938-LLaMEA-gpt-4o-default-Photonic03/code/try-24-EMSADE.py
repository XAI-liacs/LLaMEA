import numpy as np

class EMSADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F_base = 0.5
        self.CR_base = 0.9
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
                
                # Adaptive parameter adjustment
                F = self.F_base + np.random.rand() * 0.3  # F ranges from 0.5 to 0.8
                CR = max(self.CR_base - np.std(fitness) / np.mean(fitness), 0.1)  # dynamic CR based on population diversity

                if np.random.rand() < 0.5:
                    # DE/rand/1 strategy
                    mutant = x0 + F * (x1 - x2)
                else:
                    # DE/best/1 strategy
                    mutant = best_global + F * (x1 - x2)

                mutant = np.clip(mutant, lb, ub)

                cross_points = np.random.rand(self.dim) < CR
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

            # Diversity mechanism
            if evaluations % (self.population_size * 10) == 0:
                diversity_metric = np.std(pop, axis=0)
                if np.min(diversity_metric) < 1e-3:  # if diversity is too low
                    pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
                    fitness = np.array([func(x) for x in pop])
                    best_idx = np.argmin(fitness)
                    best_global = pop[best_idx]
                    evaluations += self.population_size

            pop = next_pop
            self.history.append(best_global)

        return best_global
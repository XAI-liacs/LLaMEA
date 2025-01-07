import numpy as np

class MultilevelAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        # Initialize F and CR with a broader range to allow more adaptability
        self.F = np.random.uniform(0.4, 0.9, self.population_size)
        self.CR = np.random.uniform(0.7, 1.0, self.population_size)
        self.history = []
        self.diversity_factor = 0.2  # Initial diversity factor
        self.success_archive = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        best_idx = np.argmin(fitness)
        best_global = pop[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            next_pop = np.zeros_like(pop)
            next_F = np.zeros_like(self.F)
            next_CR = np.zeros_like(self.CR)

            for i in range(self.population_size):
                indices = np.random.choice(range(self.population_size), 3, replace=False)
                x0, x1, x2 = pop[indices]
                
                # DE/rand/1 strategy with adaptive parameters
                mutant = x0 + self.F[i] * (x1 - x2)
                mutant = np.clip(mutant, lb, ub)

                cross_points = np.random.rand(self.dim) < self.CR[i]
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
                    self.success_archive.append((self.F[i], self.CR[i]))
                else:
                    next_pop[i] = pop[i]

                # Multilevel adaptation of F and CR
                if len(self.success_archive) > 0:
                    self.F[i], self.CR[i] = self._adapt_parameters()

            pop = next_pop
            self.history.append(best_global)

        return best_global

    def _adapt_parameters(self):
        mean_F = np.mean([f for f, _ in self.success_archive])
        mean_CR = np.mean([cr for _, cr in self.success_archive])
        # Random perturbation around historical mean
        new_F = np.clip(mean_F + 0.1 * (np.random.rand() - 0.5), 0.4, 0.9)
        new_CR = np.clip(mean_CR + 0.1 * (np.random.rand() - 0.5), 0.7, 1.0)
        return new_F, new_CR
import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = 0.5   # Initial scaling factor
        self.CR = 0.9  # Initial crossover rate
        self.success_history = [0.5, 0.5, 0.5]  # Three strategies
        self.diversity_factor = 0.2  # Initial diversity factor

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        best_idx = np.argmin(fitness)
        best_global = pop[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            next_pop = np.zeros_like(pop)
            successes = [0, 0, 0]  # Successes for each strategy

            for i in range(self.population_size):
                indices = np.random.choice(range(self.population_size), 3, replace=False)
                x0, x1, x2 = pop[indices]
                
                # Select strategy based on success history
                strategy = np.random.choice([0, 1, 2], p=np.array(self.success_history) / sum(self.success_history))
                if strategy == 0:
                    # DE/rand/1 with noise
                    noise = np.random.uniform(-self.diversity_factor, self.diversity_factor, self.dim)
                    mutant = x0 + self.F * (x1 - x2) + noise
                elif strategy == 1:
                    # DE/best/1
                    mutant = best_global + self.F * (x1 - x2)
                else:
                    # DE/current-to-best/1
                    mutant = x0 + self.F * (best_global - x0) + self.F * (x1 - x2)

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
                    successes[strategy] += 1
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best_global = trial
                else:
                    next_pop[i] = pop[i]

            # Update success history for strategies
            total_successes = sum(successes)
            if total_successes > 0:
                self.success_history = [s / total_successes for s in successes]

            # Adapt F, CR, and diversity factor based on success
            self.F = np.clip(self.F + 0.1 * (np.random.rand() - 0.5), 0.4, 0.9)
            self.CR = np.clip(self.CR + 0.1 * (np.random.rand() - 0.5), 0.7, 1.0)
            self.diversity_factor = np.clip(self.diversity_factor + 0.05 * (np.random.rand() - 0.5), 0.1, 0.3)

            pop = next_pop

        return best_global
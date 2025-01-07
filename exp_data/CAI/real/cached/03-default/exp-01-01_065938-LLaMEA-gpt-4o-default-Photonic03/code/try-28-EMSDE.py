import numpy as np

class EMSDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = 0.5   # Initial scaling factor
        self.CR = 0.9  # Initial crossover rate
        self.success_rates = [0.5, 0.5] # To track success of strategies
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
            successes = [0, 0]  # To track the number of successes for each strategy

            for i in range(self.population_size):
                indices = np.random.choice(range(self.population_size), 3, replace=False)
                x0, x1, x2 = pop[indices]
                
                # Select strategy based on success rates
                strategy = np.random.choice([0, 1], p=self.success_rates)
                if strategy == 0:
                    mutant = x0 + self.F * (x1 - x2)  # DE/rand/1 strategy
                else:
                    mutant = best_global + self.F * (x1 - x2)  # DE/best/1 strategy

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

            # Update success rates for strategies
            total_successes = sum(successes)
            if total_successes > 0:
                self.success_rates = [s / total_successes for s in successes]

            # Adapt F and CR based on success or failure
            self.F = np.clip(self.F + 0.1 * (np.random.rand() - 0.5), 0.4, 0.9)
            self.CR = np.clip(self.CR + 0.1 * (np.random.rand() - 0.5), 0.7, 1.0)

            pop = next_pop
            self.history.append(best_global)

        return best_global
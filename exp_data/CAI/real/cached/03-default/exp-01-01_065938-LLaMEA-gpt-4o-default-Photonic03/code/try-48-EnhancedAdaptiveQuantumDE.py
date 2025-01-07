import numpy as np

class EnhancedAdaptiveQuantumDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.base_population_size = 50
        self.F = 0.5
        self.CR = 0.9
        self.success_rates = [0.5, 0.5]
        self.history = []
        self.diversity_factor = 0.2

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.base_population_size
        population_quantum = np.random.uniform(0, 1, (population_size, self.dim))
        pop = lb + (ub - lb) * np.sin(np.pi * population_quantum)
        fitness = np.array([func(x) for x in pop])
        best_idx = np.argmin(fitness)
        best_global = pop[best_idx]

        evaluations = population_size

        while evaluations < self.budget:
            next_pop = np.zeros_like(pop)
            successes = [0, 0]

            for i in range(population_size):
                indices = np.random.choice(range(population_size), 3, replace=False)
                x0, x1, x2 = pop[indices]
                
                strategy = np.random.choice([0, 1], p=self.success_rates)
                if strategy == 0:
                    noise = np.random.uniform(-self.diversity_factor, self.diversity_factor, self.dim)
                    mutant = x0 + self.F * (x1 - x2) + noise
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
                    successes[strategy] += 1
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best_global = trial
                else:
                    next_pop[i] = pop[i]

            total_successes = sum(successes)
            if total_successes > 0:
                self.success_rates = [s / total_successes for s in successes]

            self.F = np.clip(self.F + 0.1 * (np.random.rand() - 0.5), 0.4, 0.9)
            self.CR = np.clip(self.CR + 0.1 * (np.random.rand() - 0.5), 0.7, 1.0)
            self.diversity_factor = np.clip(self.diversity_factor + 0.05 * (np.random.rand() - 0.5), 0.1, 0.3)

            # Dynamic adjustment of population size
            if evaluations < self.budget / 2:
                population_size = min(self.base_population_size + (evaluations // 100), self.budget - evaluations)
            else:
                population_size = max(self.base_population_size, self.base_population_size - (evaluations // 100))

            pop = next_pop[:population_size]
            fitness = fitness[:population_size]
            self.history.append(best_global)

        return best_global
import numpy as np

class QuantumAssistedSelfAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = np.full(self.population_size, 0.5)
        self.CR = np.full(self.population_size, 0.9)
        self.success_rates = [0.5, 0.5]
        self.history = []
        self.diversity_factor = 0.2
        self.tau1 = 0.1  # Probability for adapting F
        self.tau2 = 0.1  # Probability for adapting CR

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_quantum = np.random.uniform(0, 1, (self.population_size, self.dim))
        pop = lb + (ub - lb) * np.sin(np.pi * population_quantum)
        fitness = np.array([func(x) for x in pop])
        best_idx = np.argmin(fitness)
        best_global = pop[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            next_pop = np.zeros_like(pop)
            successes = [0, 0]

            for i in range(self.population_size):
                if np.random.rand() < self.tau1:
                    self.F[i] = np.random.uniform(0.4, 0.9)
                if np.random.rand() < self.tau2:
                    self.CR[i] = np.random.uniform(0.7, 1.0)

                indices = np.random.choice(range(self.population_size), 3, replace=False)
                x0, x1, x2 = pop[indices]

                strategy = np.random.choice([0, 1], p=self.success_rates)
                if strategy == 0:
                    noise = np.random.uniform(-self.diversity_factor, self.diversity_factor, self.dim)
                    mutant = x0 + self.F[i] * (x1 - x2) + noise
                else:
                    mutant = best_global + self.F[i] * (x1 - x2)

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
                    successes[strategy] += 1
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best_global = trial
                else:
                    next_pop[i] = pop[i]

            total_successes = sum(successes)
            if total_successes > 0:
                self.success_rates = [s / total_successes for s in successes]

            self.diversity_factor = np.clip(self.diversity_factor + 0.05 * (np.random.rand() - 0.5), 0.1, 0.3)

            pop = next_pop
            self.history.append(best_global)

        return best_global
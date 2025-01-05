import numpy as np

class EnhancedQuantumInspiredLevyFlightDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F_min, self.F_max = 0.4, 0.9
        self.CR_min, self.CR_max = 0.7, 1.0
        self.diversity_factor_min, self.diversity_factor_max = 0.1, 0.3
        self.success_rates = [0.5, 0.5]
        self.history = []

    def levy_flight(self, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v) ** (1 / beta)
        return 0.01 * step

    def adapt_population_size(self, evaluations):
        return max(20, int(self.population_size * (1 - evaluations / self.budget)))

    def exponential_crossover(self, target, mutant, CR):
        n = len(target)
        j_rand = np.random.randint(0, n)
        cross_points = np.zeros(n, dtype=bool)
        for j in range(n):
            if np.random.rand() < CR or j == j_rand:
                cross_points[j] = True
        return np.where(cross_points, mutant, target)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_quantum = np.random.uniform(0, 1, (self.population_size, self.dim))
        pop = lb + (ub - lb) * np.cos(np.pi * population_quantum)
        fitness = np.array([func(x) for x in pop])
        best_idx = np.argmin(fitness)
        best_global = pop[best_idx]
        historical_best_global = best_global

        evaluations = self.population_size

        while evaluations < self.budget:
            cur_population_size = self.adapt_population_size(evaluations)
            next_pop = np.zeros((cur_population_size, self.dim))
            successes = [0, 0]

            for i in range(cur_population_size):
                indices = np.random.choice(range(len(pop)), 3, replace=False)
                x0, x1, x2 = pop[indices]
                
                strategy = np.random.choice([0, 1], p=self.success_rates)
                F = np.random.uniform(self.F_min, self.F_max)
                CR = np.random.uniform(self.CR_min, self.CR_max)
                diversity_factor = np.random.uniform(self.diversity_factor_min, self.diversity_factor_max)

                levy_step = self.levy_flight()
                
                if strategy == 0:
                    noise = np.random.uniform(-diversity_factor, diversity_factor, self.dim) + levy_step
                    mutant = x0 + F * (x1 - x2) + noise
                else:
                    mutant = historical_best_global + F * (x1 - x2) + levy_step

                mutant = np.clip(mutant, lb, ub)

                trial = self.exponential_crossover(pop[i % len(pop)], mutant, CR)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i % len(pop)]:
                    next_pop[i] = trial
                    fitness[i % len(pop)] = trial_fitness
                    successes[strategy] += 1
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i % len(pop)
                        best_global = trial
                        if trial_fitness < func(historical_best_global):
                            historical_best_global = trial
                else:
                    next_pop[i] = pop[i % len(pop)]

            total_successes = sum(successes)
            if total_successes > 0:
                self.success_rates = [s / total_successes for s in successes]

            self.history.append(historical_best_global)
            pop = next_pop

        return historical_best_global
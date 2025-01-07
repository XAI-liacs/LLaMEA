import numpy as np

class QuantumInspiredAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F_base = 0.5
        self.CR_min, self.CR_max = 0.7, 1.0
        self.success_rates = [0.5, 0.5]
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_quantum = np.random.uniform(0, 1, (self.population_size, self.dim))
        pop = lb + (ub - lb) * np.cos(np.pi * population_quantum)
        fitness = np.array([func(x) for x in pop])
        best_idx = np.argmin(fitness)
        best_global = pop[best_idx]
        evaluations = self.population_size
        last_improvement = 0

        while evaluations < self.budget:
            next_pop = np.zeros_like(pop)
            successes = [0, 0]
            progress = evaluations / self.budget
            F_dynamic = self.F_base + (1 - progress) * (1 - self.F_base)  # Dynamic scaling of F

            for i in range(self.population_size):
                indices = np.random.choice(range(self.population_size), 3, replace=False)
                x0, x1, x2 = pop[indices]
                
                strategy = np.random.choice([0, 1], p=self.success_rates)
                F = np.random.uniform(self.F_base, F_dynamic)
                CR = np.random.uniform(self.CR_min, self.CR_max)

                if strategy == 0:
                    mutant = x0 + F * (x1 - x2)
                else:
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
                    successes[strategy] += 1
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best_global = trial
                        last_improvement = evaluations
                else:
                    next_pop[i] = pop[i]

            total_successes = sum(successes)
            if total_successes > 0:
                self.success_rates = [s / total_successes for s in successes]

            if evaluations - last_improvement > self.population_size:
                break  # Terminate early if no improvement

            self.history.append(best_global)
            pop = next_pop

        return best_global
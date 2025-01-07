import numpy as np

class HybridDEAnnealingOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                (np.math.gamma((1 + beta) / 2) * beta * 
                2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(size) * sigma
        v = np.random.randn(size)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def __call__(self, func):
        pop_size = min(30 + int(20 * np.random.rand()), self.budget // 10)  # Dynamic population size
        F = 0.85
        CR = 0.9
        temp_initial = 1000.0
        temp_min = 1e-5
        cooling_rate = 0.95

        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        pop_fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(pop_fitness)
        best_solution = pop[best_idx]
        best_fitness = pop_fitness[best_idx]

        evaluations = pop_size

        while evaluations < self.budget and temp_initial > temp_min:
            for i in range(pop_size):
                if evaluations >= self.budget:
                    break

                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                F_adaptive = 0.5 + 0.5 * (1 - evaluations / self.budget) * np.random.rand()
                mutant = np.clip(a + F_adaptive * (b - c) + self.levy_flight(self.dim), lb, ub)  # Apply Lévy flight

                CR_adaptive = 0.7 + 0.3 * np.random.rand()
                crossover_mask = np.random.rand(self.dim) < CR_adaptive
                trial = np.where(crossover_mask, mutant, pop[i])

                trial_fitness = func(trial)
                evaluations += 1

                if (trial_fitness < pop_fitness[i]) or (np.random.rand() < np.exp((pop_fitness[i] - trial_fitness) / temp_initial)):
                    pop[i] = trial
                    pop_fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

            temp_initial *= cooling_rate * (1 - evaluations / self.budget)

        return best_solution, best_fitness
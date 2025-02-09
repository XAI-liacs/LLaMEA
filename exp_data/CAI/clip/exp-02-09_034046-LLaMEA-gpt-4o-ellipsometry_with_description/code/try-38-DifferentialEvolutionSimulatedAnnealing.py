import numpy as np

class DifferentialEvolutionSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 10 * self.dim
        sub_pop_size = pop_size // 2  # Multi-population strategy
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        best_fitness = fitness[best_idx]
        
        CR = 0.9
        F = np.full(pop_size, 0.8)
        temperature = 1.0

        for generation in range(self.budget - pop_size):
            for sp_start in range(0, pop_size, sub_pop_size):
                sub_pop = pop[sp_start:sp_start + sub_pop_size]
                sub_fitness = fitness[sp_start:sp_start + sub_pop_size]
                local_best_idx = np.argmin(sub_fitness)
                local_best = sub_pop[local_best_idx]

                for i in range(sp_start, sp_start + sub_pop_size):
                    idxs = [idx for idx in range(sp_start, sp_start + sub_pop_size) if idx != i]
                    a, b, c = sub_pop[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + F[i] * (b - c), func.bounds.lb, func.bounds.ub)
                    
                    CR = 0.6 + (0.4 * np.abs(sub_fitness[i-sp_start] - local_best_idx) / (np.max(sub_fitness) - np.min(sub_fitness)))

                    cross_points = np.random.rand(self.dim) < CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, pop[i])
                    trial_fitness = func(trial)

                    if trial_fitness < sub_fitness[i - sp_start] or np.random.rand() < np.exp((sub_fitness[i - sp_start] - trial_fitness) / temperature):
                        pop[i] = trial
                        fitness[i] = trial_fitness
                        if trial_fitness < best_fitness:
                            best_fitness = trial_fitness
                            best = trial

            temperature *= 0.98 - 0.02 * (generation / self.budget)

        return best, best_fitness
import numpy as np

class DynamicEnsembleDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 12 * self.dim  # Adjusted population size for diversity
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        best_fitness = fitness[best_idx]
        
        CR = 0.9
        F = np.full(pop_size, 0.7)  # Adjusted starting differential weight
        temperature = 1.0

        for generation in range(self.budget - pop_size):
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F[i] * (b - c), func.bounds.lb, func.bounds.ub)

                CR = 0.5 + (0.5 * (1 - (generation / (self.budget * 0.4))))  # Different CR evolution

                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = func(trial)

                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temperature) and trial_fitness < 0.93 * best_fitness:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best = trial

                diversity = np.std(pop, axis=0).mean()
                F[i] = 0.5 + 0.25 * (best_fitness - fitness[i]) / (1 + diversity)
                F[i] = np.clip(F[i], 0.4, 1.0)

            temperature *= 0.97 - 0.03 * (generation / self.budget)
            if generation % int(self.budget * 0.1) == 0 and generation > 0:
                pop_size = min(pop_size + 1, self.budget - generation)

        return best, best_fitness
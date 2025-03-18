import numpy as np
from scipy.stats import levy

class DifferentialEvolutionSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 10 * self.dim
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        best_fitness = fitness[best_idx]
        
        CR = 0.9  # Crossover probability
        F = np.full(pop_size, 0.8)  # Differential weight vector
        temperature = 1.0

        for generation in range(self.budget - pop_size):
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F[i] * (b - c), func.bounds.lb, func.bounds.ub)
                
                if np.random.rand() < 0.1:  # Introduce Levy Flight with 10% probability
                    mutant += levy.rvs(size=self.dim)

                CR = 0.6 + (0.4 * np.abs(fitness[i] - best_fitness) / (np.max(fitness) - np.min(fitness)))

                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = func(trial)

                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temperature):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best = trial
                
                diversity = np.std(pop, axis=0).mean()
                F[i] = 0.5 + 0.3 * (best_fitness - fitness[i]) / (1 + diversity)

            temperature *= 0.98 - 0.02 * (generation / self.budget)

        return best, best_fitness
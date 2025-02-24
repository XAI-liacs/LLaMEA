import numpy as np

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
        
        CR = 0.9
        F = np.full(pop_size, 0.8)
        temperature = 1.0
        base_F = 0.5

        for generation in range(self.budget - pop_size):
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F[i] * (b - c), func.bounds.lb, func.bounds.ub)
                
                CR = 0.5 + (0.5 * np.cos(np.pi * generation / self.budget))  # Dynamic CR adjustment

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
                F[i] = base_F + 0.3 * (best_fitness - fitness[i]) / (1 + diversity)  # Adaptive mutation
                F[i] = np.clip(F[i], 0.4, 1.2)  # Adjust mutation range

            temperature *= 0.97 - 0.02 * (generation / self.budget)
            if generation % (self.budget // 10) == 0 and generation > 0:  # Dynamic population size adjustment
                pop_size = min(pop_size + 2, self.budget - generation)

        return best, best_fitness
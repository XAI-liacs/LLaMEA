import numpy as np

class DifferentialEvolutionSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        initial_pop_size = 10 * self.dim
        pop_size = initial_pop_size
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        best_fitness = fitness[best_idx]
        
        CR = 0.9
        F = np.full(pop_size, 0.8)
        temperature = 1.0

        for generation in range(self.budget - initial_pop_size):
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F[i] * (b - c), func.bounds.lb, func.bounds.ub)
                if np.random.rand() < 0.5:
                    F[i] = 0.7 + 0.3 * np.random.rand()  # Dynamic mutability
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
                        CR = 0.5 + 0.5 * (best_fitness / (best_fitness + trial_fitness))
                        
                F[i] = 0.5 + 0.5 * (best_fitness - fitness[i]) / (1 + np.abs(best_fitness - fitness[i]))

            temperature *= 0.98
            if generation % (self.budget // 10) == 0 and pop_size < 20 * self.dim:  # Adaptive population size
                new_pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.dim, self.dim))
                new_fitness = np.array([func(ind) for ind in new_pop])
                pop = np.vstack((pop, new_pop))
                fitness = np.hstack((fitness, new_fitness))
                pop_size = len(pop)

        return best, best_fitness
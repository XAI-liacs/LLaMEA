import numpy as np

class AdaptiveDifferentialEvolutionSimulatedAnnealing:
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
        
        CR = np.full(pop_size, 0.9)  # Crossover probabilities per individual
        F = np.full(pop_size, 0.8)   # Differential weights per individual
        temperature = 1.0
        memory = np.zeros(pop_size)  # Memory to track improvements

        for generation in range(self.budget - pop_size):
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F[i] * (b - c), func.bounds.lb, func.bounds.ub)
                cross_points = np.random.rand(self.dim) < CR[i]
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
                    memory[i] = 1  # Mark improvement

            temperature *= 0.98  # Cool down the temperature

            # Update CR and F based on improvements
            for i in range(pop_size):
                if memory[i] == 1:
                    CR[i] = min(1, CR[i] + 0.05)
                    F[i] = np.clip(F[i] + 0.05 * (np.random.rand() - 0.5), 0.5, 1.0)
                else:
                    CR[i] = max(0.1, CR[i] - 0.05)
                    F[i] = np.clip(F[i] - 0.05 * (np.random.rand() - 0.5), 0.5, 1.0)
                memory[i] = 0  # Reset memory

        return best, best_fitness
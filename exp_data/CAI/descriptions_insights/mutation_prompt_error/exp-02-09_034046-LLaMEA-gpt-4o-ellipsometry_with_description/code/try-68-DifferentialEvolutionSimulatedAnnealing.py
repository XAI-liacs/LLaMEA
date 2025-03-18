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
        
        CR = 0.9  # Crossover probability
        temperature = 1.0
        
        adaptive_F = np.linspace(0.5, 1.0, pop_size)

        for generation in range(self.budget - pop_size):
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                
                # Fitness-based mutation strategy
                weighted_diff = adaptive_F[i] * (b - c)
                mutant = np.clip(a + weighted_diff, func.bounds.lb, func.bounds.ub)

                CR = 0.2 + (0.7 * (1 - (generation / self.budget)))  # Dynamic CR adjustment

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

            temperature *= 0.98 - 0.02 * (generation / self.budget)

            # Update adaptive_F based on current generation
            adaptive_F = 0.5 + 0.5 * np.exp(-np.linspace(0, 1, pop_size) * (generation / self.budget))

        return best, best_fitness
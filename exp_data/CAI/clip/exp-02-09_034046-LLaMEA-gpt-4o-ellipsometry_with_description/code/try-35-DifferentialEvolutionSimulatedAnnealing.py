import numpy as np

class DifferentialEvolutionSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        num_subpops = 3  # Using multiple subpopulations
        subpop_size = (10 * self.dim) // num_subpops
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (num_subpops * subpop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        best_fitness = fitness[best_idx]
        
        CR = 0.9  # Crossover probability
        F = np.full(num_subpops * subpop_size, 0.8)  # Differential weight vector
        temperature = 1.0

        for generation in range(self.budget - num_subpops * subpop_size):
            for subpop in range(num_subpops):
                for i in range(subpop * subpop_size, (subpop + 1) * subpop_size):
                    idxs = [idx for idx in range(subpop * subpop_size, (subpop + 1) * subpop_size) if idx != i]
                    a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                    F[i] = 0.5 + 0.4 * np.random.rand()  # Adaptive mutation scaling
                    mutant = np.clip(a + F[i] * (b - c), func.bounds.lb, func.bounds.ub)
                    
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

        return best, best_fitness
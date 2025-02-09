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
        F = 0.8   # Differential weight
        temperature = 1.0

        for generation in range(self.budget - pop_size):
            # Line change 1: Reduce population size adaptively
            if generation % 50 == 0 and pop_size > 2:
                pop_size = max(2, pop_size - 1)
                pop = pop[:pop_size]
                fitness = fitness[:pop_size]
            
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)
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
                        CR = 0.5 + 0.5 * (best_fitness / (best_fitness + trial_fitness))  # Adapt CR
                        F = 0.5 + 0.5 * np.abs(best_fitness - trial_fitness)  # Adapt F

            # Line change 2: Enhance the cooling schedule
            temperature *= 0.995  # Cool down the temperature

        return best, best_fitness
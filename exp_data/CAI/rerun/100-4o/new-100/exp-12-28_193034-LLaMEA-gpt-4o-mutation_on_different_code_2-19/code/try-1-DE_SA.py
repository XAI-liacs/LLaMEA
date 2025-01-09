import numpy as np

class DE_SA:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.8, CR=0.9, init_temp=1.0, alpha=0.99):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.init_temp = init_temp
        self.alpha = alpha
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        # Set initial best solution
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        temp = self.init_temp
        evaluations = self.pop_size
        
        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Select three distinct indices
                indices = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                a, b, c = population[indices]

                # Differential Evolution mutation and crossover
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                trial = np.copy(population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Evaluate trial solution
                f_trial = func(trial)
                evaluations += 1

                # Simulated Annealing acceptance criterion
                if f_trial < fitness[i] or np.random.rand() < np.exp((fitness[i] - f_trial) / temp):
                    population[i] = trial
                    fitness[i] = f_trial

                # Update best solution
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if evaluations >= self.budget:
                    break

            # Decrease temperature quadratically
            temp *= self.alpha**2

        return self.f_opt, self.x_opt
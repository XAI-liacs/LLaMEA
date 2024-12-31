import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.F = 0.5  # Initial mutation factor
        self.CR = 0.9  # Initial crossover rate

    def __call__(self, func):
        # Initialize population within bounds
        bounds = func.bounds
        population = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for i in range(self.budget - self.pop_size):
            for j in range(self.pop_size):
                # Mutation: DE/rand/1 scheme
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + self.F * (x2 - x3)
                mutant = np.clip(mutant, bounds.lb, bounds.ub)

                # Crossover
                trial = np.copy(population[j])
                crossover_points = np.random.rand(self.dim) < self.CR
                if not np.any(crossover_points):
                    crossover_points[np.random.randint(0, self.dim)] = True
                trial[crossover_points] = mutant[crossover_points]

                # Selection
                f_trial = func(trial)
                if f_trial < fitness[j]:
                    fitness[j] = f_trial
                    population[j] = trial

                    # Update best solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

            # Dynamic parameter control
            success_rate = np.mean(fitness < self.f_opt)
            self.F = 0.5 + 0.1 * (1 - success_rate)  # Adaptive mutation factor
            self.CR = 0.9 - 0.2 * success_rate  # Adaptive crossover rate

        return self.f_opt, self.x_opt
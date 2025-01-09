import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20, F_min=0.4, F_max=0.9, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F_min = F_min  # Minimum mutation factor
        self.F_max = F_max  # Maximum mutation factor
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        evals = self.pop_size  # Initial evaluations done

        while evals < self.budget:
            for i in range(self.pop_size):
                F = np.random.uniform(self.F_min, self.F_max)  # Adaptive mutation factor
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + F * (x2 - x3), bounds[:, 0], bounds[:, 1])

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                f_trial = func(trial)
                evals += 1
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if evals >= self.budget:
                    break

            # Elitism: carry the best individual to the next generation
            best_idx = np.argmin(fitness)
            population[0] = population[best_idx]
            fitness[0] = fitness[best_idx]
            
        return self.f_opt, self.x_opt
import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20 * dim  # Population size is typically 10-20 times the dimension
        self.f_opt = np.Inf
        self.x_opt = None
        self.F = 0.5  # Initial mutation factor
        self.CR = 0.9  # Initial crossover rate

    def __call__(self, func):
        # Initialize population
        bounds = (func.bounds.lb, func.bounds.ub)
        pop = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        # Track the best solution
        self.f_opt = fitness.min()
        self.x_opt = pop[fitness.argmin()]

        eval_count = self.pop_size

        while eval_count < self.budget:
            for i in range(self.pop_size):
                # Mutation: DE/rand/1/bin
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds[0], bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                f = func(trial)
                eval_count += 1

                if f < fitness[i]:
                    fitness[i] = f
                    pop[i] = trial

                    # Update best known solution
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial

                    # Adaptive strategies
                    self.F = np.clip(self.F + 0.1 * np.random.randn(), 0.1, 1.0)
                    self.CR = np.clip(self.CR + 0.1 * np.random.randn(), 0.1, 1.0)

            if eval_count >= self.budget:
                break

        return self.f_opt, self.x_opt
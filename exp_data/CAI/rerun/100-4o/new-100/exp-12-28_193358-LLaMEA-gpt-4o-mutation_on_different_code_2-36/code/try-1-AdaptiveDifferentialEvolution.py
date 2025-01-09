import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.F = 0.5  # Initial mutation factor
        self.CR = 0.9 # Initial crossover rate

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        pop = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                indices = [ind for ind in range(self.population_size) if ind != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                
                mutant = np.clip(a + self.F * (b - c), bounds[0], bounds[1])
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                f_trial = func(trial)
                
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

            # Adapt F and CR based on refined success rate
            success_rate = np.mean(fitness < np.min(fitness))
            self.F = 0.5 + 0.5 * np.random.rand() if success_rate > 0.2 else 0.5
            self.CR = 0.9 * np.random.rand() if success_rate < 0.2 else 0.9

        return self.f_opt, self.x_opt
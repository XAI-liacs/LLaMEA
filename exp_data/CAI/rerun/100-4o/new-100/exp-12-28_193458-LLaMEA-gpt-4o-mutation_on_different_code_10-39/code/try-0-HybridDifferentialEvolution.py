import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 100
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        lower_bound, upper_bound = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        pop_fitness = np.array([func(ind) for ind in pop])
        
        self.f_opt = np.min(pop_fitness)
        self.x_opt = pop[np.argmin(pop_fitness)]

        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                
                # Mutation
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = pop[idxs]
                mutant = np.clip(a + self.F * (b - c), lower_bound, upper_bound)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Local Search
                local_trial = self.local_search(trial, func, lower_bound, upper_bound)
                
                # Selection
                f_trial = func(local_trial)
                eval_count += 1
                if f_trial < pop_fitness[i]:
                    pop[i] = local_trial
                    pop_fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = local_trial
        
        return self.f_opt, self.x_opt

    def local_search(self, x, func, lb, ub):
        """ Simple local search around x """
        step_size = 0.1 * (ub - lb)
        for _ in range(5):  # Limited local search steps
            neighbor = np.clip(x + np.random.uniform(-step_size, step_size), lb, ub)
            if func(neighbor) < func(x):
                x = neighbor
        return x
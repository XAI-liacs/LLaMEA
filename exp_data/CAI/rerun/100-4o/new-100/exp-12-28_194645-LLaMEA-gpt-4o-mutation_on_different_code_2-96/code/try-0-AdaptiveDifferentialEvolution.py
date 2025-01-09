import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = max(10, dim * 5)
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.dynamic_pop_shrink = 0.99

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        for eval_count in range(self.budget):
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                f_trial = func(trial)
                eval_count += 1
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                if eval_count >= self.budget:
                    break

            # Dynamically adjust population size
            self.population_size = int(max(5, self.population_size * self.dynamic_pop_shrink))
            pop = pop[:self.population_size]
            fitness = fitness[:self.population_size]

        return self.f_opt, self.x_opt
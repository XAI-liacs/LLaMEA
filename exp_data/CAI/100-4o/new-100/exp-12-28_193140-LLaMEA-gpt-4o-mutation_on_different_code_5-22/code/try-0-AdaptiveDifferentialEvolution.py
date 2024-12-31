import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=30):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        def mutate(target_idx, scale_factor=0.8):
            idxs = [idx for idx in range(self.population_size) if idx != target_idx]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + scale_factor * (b - c), lb, ub)
            return mutant

        def crossover(target, mutant, crossover_rate=0.9):
            cross_points = np.random.rand(self.dim) < crossover_rate
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, target)
            return trial

        evals = self.population_size
        while evals < self.budget:
            for i in range(self.population_size):
                mutant = mutate(i, scale_factor=np.random.uniform(0.5, 1.0))
                trial = crossover(population[i], mutant, crossover_rate=np.random.uniform(0.5, 1.0))
                f = func(trial)
                evals += 1

                if f < fitness[i]:
                    population[i], fitness[i] = trial, f
                    if f < self.f_opt:
                        self.f_opt, self.x_opt = f, trial
                
                if evals >= self.budget:
                    break

        return self.f_opt, self.x_opt
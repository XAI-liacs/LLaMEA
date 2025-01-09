import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        evals = self.pop_size
        last_improvement = 0
        
        while evals < self.budget:
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds[0], bounds[1])
                
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[i])
                
                f_trial = func(trial)
                evals += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        last_improvement = evals
                
                if evals % (self.pop_size * 10) == 0:
                    f_mean = fitness.mean()
                    if self.f_opt < f_mean:
                        self.F = min(self.F + 0.1, 1.0)
                        self.CR = max(self.CR - 0.1, 0.1)
                    else:
                        self.F = max(self.F - 0.1, 0.1)
                        self.CR = min(self.CR + 0.1, 1.0)
                    
                    diversity = np.mean(np.std(population, axis=0))
                    if diversity < 0.1 and evals - last_improvement > self.pop_size * 5:
                        self.pop_size = max(5, self.pop_size - 2)
                        population = population[:self.pop_size]
                        fitness = fitness[:self.pop_size]
                    elif diversity > 0.2 and evals - last_improvement < self.pop_size * 2:
                        self.pop_size = min(50, self.pop_size + 2)
                        new_pop = np.random.uniform(bounds[0], bounds[1], (2, self.dim))
                        new_fitness = np.array([func(ind) for ind in new_pop])
                        population = np.vstack((population, new_pop))
                        fitness = np.hstack((fitness, new_fitness))
                        evals += 2

        return self.f_opt, self.x_opt
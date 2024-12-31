import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = [func.bounds.lb, func.bounds.ub]
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        eval_count = self.pop_size
        while eval_count < self.budget:
            diversity = np.mean(np.std(population, axis=0))
            adaptive_CR = self.CR * (1 - diversity / np.max((diversity, 1e-8))) + 0.1

            for i in range(self.pop_size):
                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = population[indices]
                
                mutant = np.clip(a + self.F * (b - c), bounds[0], bounds[1])
                
                crossover = np.random.rand(self.dim) < adaptive_CR
                trial = np.where(crossover, mutant, population[i])
                
                f_trial = func(trial)
                eval_count += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if eval_count % (self.budget // 10) == 0:
                    successful_mutations = np.sum(fitness < self.f_opt)
                    self.pop_size = min(max(5, int(0.5 * self.pop_size + 0.5 * successful_mutations)), len(population))
                    population = population[:self.pop_size]
                    fitness = fitness[:self.pop_size]

                if eval_count >= self.budget:
                    break

        return self.f_opt, self.x_opt
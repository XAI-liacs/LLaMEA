import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.inf
        self.x_opt = None

    def _initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def _mutate(self, idx, population):
        indices = [i for i in range(self.pop_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = population[a] + self.F * (population[b] - population[c])
        return mutant

    def _crossover(self, target, mutant):
        trial = np.copy(target)
        for i in range(self.dim):
            if np.random.rand() < self.CR or i == np.random.randint(self.dim):
                trial[i] = mutant[i]
        return trial

    def _adapt_params(self, diversity):
        self.F = 0.5 + 0.3 * np.exp(-diversity)
        self.CR = 0.9 * np.exp(-diversity)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self._initialize_population(lb, ub)
        fitness = np.array([func(ind) for ind in population])
        
        eval_count = self.pop_size
        
        while eval_count < self.budget:
            diversity = np.std(population, axis=0).mean()
            self._adapt_params(diversity)
            
            for i in range(self.pop_size):
                mutant = self._mutate(i, population)
                trial = self._crossover(population[i], mutant)
                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                eval_count += 1
                
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                
                if eval_count >= self.budget:
                    break
        
        return self.f_opt, self.x_opt
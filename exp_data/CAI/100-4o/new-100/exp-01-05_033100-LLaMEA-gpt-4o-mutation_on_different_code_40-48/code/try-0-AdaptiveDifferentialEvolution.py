import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.F = 0.5  # Mutation factor
        self.CR = 0.9  # Crossover rate

    def _mutate(self, population, target_idx):
        indices = list(range(self.pop_size))
        indices.remove(target_idx)
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant = a + self.F * (b - c)
        return np.clip(mutant, -5.0, 5.0)

    def _crossover(self, target, mutant):
        crossover = np.random.rand(self.dim) < self.CR
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover, mutant, target)
        return trial

    def _adaptive_control(self, success_rate):
        if success_rate < 0.2:
            self.F *= 0.8
            self.CR *= 0.9
        elif success_rate > 0.8:
            self.F = min(self.F * 1.2, 1.0)
            self.CR = min(self.CR * 1.1, 1.0)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for i in range(self.budget - self.pop_size):
            target_idx = i % self.pop_size
            target = population[target_idx]
            mutant = self._mutate(population, target_idx)
            trial = self._crossover(target, mutant)
            f_trial = func(trial)
            
            if f_trial < fitness[target_idx]:
                fitness[target_idx] = f_trial
                population[target_idx] = trial
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
            
            success_rate = np.mean(fitness < self.f_opt)
            self._adaptive_control(success_rate)
        
        return self.f_opt, self.x_opt
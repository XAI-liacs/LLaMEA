import numpy as np

class DAD_ES:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20
        self.scaling_factor = 0.5
        self.crossover_prob = 0.7
        self.bounds = (-5.0, 5.0)
    
    def _initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
    
    def _mutate(self, population, best_idx):
        indices = np.arange(self.population_size)
        for i in range(self.population_size):
            idxs = indices[indices != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            if i == best_idx:
                yield best_idx, a + self.scaling_factor * (b - c)
            else:
                yield i, a + self.scaling_factor * (b - c)
    
    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return np.clip(trial, self.bounds[0], self.bounds[1])

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        
        for evals in range(self.budget - self.population_size):
            for target_idx, mutant in self._mutate(population, best_idx):
                trial = self._crossover(population[target_idx], mutant)
                f_trial = func(trial)
                
                if f_trial < fitness[target_idx]:
                    population[target_idx] = trial
                    fitness[target_idx] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        best_idx = np.argmin(fitness)
            
            self.scaling_factor = np.random.uniform(0.4, 1.0)
            self.crossover_prob = np.random.uniform(0.6, 0.9)
        
        return self.f_opt, self.x_opt
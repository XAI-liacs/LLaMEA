import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20 * self.dim
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.bounds = (-5.0, 5.0)
        
    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
    
    def mutate(self, population, idx):
        candidates = np.random.choice(np.delete(np.arange(self.population_size), idx), 3, replace=False)
        return population[candidates[0]] + self.mutation_factor * (population[candidates[1]] - population[candidates[2]])
    
    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_prob
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        return np.where(crossover_mask, mutant, target)
    
    def bound_check(self, vector):
        return np.clip(vector, self.bounds[0], self.bounds[1])
    
    def adapt_parameters(self):
        diversity = np.std(self.population, axis=0).mean()
        self.mutation_factor = 0.5 + 0.3 * (1 - diversity)
        self.crossover_prob = 0.6 + 0.3 * diversity

    def __call__(self, func):
        self.population = self.initialize_population()
        fitness = np.array([func(ind) for ind in self.population])
        self.f_opt = np.min(fitness)
        self.x_opt = self.population[np.argmin(fitness)]
        
        for _ in range(self.budget - self.population_size):
            self.adapt_parameters()
            for i in range(self.population_size):
                mutant = self.mutate(self.population, i)
                trial = self.crossover(self.population[i], mutant)
                trial = self.bound_check(trial)
                
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    self.population[i] = trial
                    
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
        
        return self.f_opt, self.x_opt
import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.initialized = False

    def initialize_population(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        self.fitness = np.array([func(ind) for ind in self.population])
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.initialized = True

    def adapt_parameters(self, iteration):
        F = 0.5 + 0.4 * (iteration / self.budget)
        CR = 0.1 + 0.8 * (1 - iteration / self.budget)
        return F, CR

    def mutate(self, idx, F):
        indices = list(range(self.population_size))
        indices.remove(idx)
        best_idx = np.argmin(self.fitness)
        if np.random.rand() < 0.1:  # Increased exploration factor
            a, b = np.random.choice(indices, 2, replace=False)
            return np.clip(self.population[best_idx] + F * (self.population[a] - self.population[b]), self.bounds[0], self.bounds[1])
        else:
            a, b, c = np.random.choice(indices, 3, replace=False)
            return np.clip(self.population[a] + F * (self.population[b] - self.population[c]), self.bounds[0], self.bounds[1])

    def crossover(self, target, mutant, CR):
        crossover_points = np.random.rand(self.dim) < CR
        if not np.any(crossover_points):
            crossover_points[np.random.randint(0, self.dim)] = True
        offspring = np.where(crossover_points, mutant, target)
        return offspring

    def resize_population(self, iteration):
        new_size = int(self.population_size * (1.0 - 0.3 * iteration / self.budget)) + 1
        if new_size < self.population_size:
            sorted_indices = np.argsort(self.fitness)
            self.population = self.population[sorted_indices[:new_size]]
            self.fitness = self.fitness[sorted_indices[:new_size]]
            self.population_size = new_size
    
    def __call__(self, func):
        if not self.initialized:
            self.initialize_population(func)
        
        for iteration in range(self.budget - self.population_size):
            F, CR = self.adapt_parameters(iteration)
            self.resize_population(iteration)
            for i in range(self.population_size):
                mutant = self.mutate(i, F)
                trial = self.crossover(self.population[i], mutant, CR)
                trial_fitness = func(trial)
                
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

        return self.f_opt, self.x_opt
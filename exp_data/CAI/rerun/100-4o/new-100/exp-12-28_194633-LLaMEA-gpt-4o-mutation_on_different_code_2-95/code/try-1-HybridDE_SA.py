import numpy as np

class HybridDE_SA:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.F = 0.8  # DE mutation factor
        self.CR = 0.9 # Crossover probability
        self.temp = 1000  # Initial temperature for SA
        self.cooling_rate = 0.99
    
    def de_mutation(self, population, idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        return population[a] + self.F * (population[b] - population[c])
    
    def de_crossover(self, target, mutant):
        crossover = np.random.rand(self.dim) < self.CR
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover, mutant, target)
        return np.clip(trial, -5.0, 5.0)
    
    def simulated_annealing(self, current, candidate, f_current, f_candidate):
        if f_candidate < f_current:
            return candidate, f_candidate
        else:
            acceptance_prob = np.exp((f_current - f_candidate) / self.temp)
            return (candidate, f_candidate) if np.random.rand() < acceptance_prob else (current, f_current)
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                mutant = self.de_mutation(population, i)
                trial = self.de_crossover(population[i], mutant)
                f_trial = func(trial)
                evaluations += 1
                population[i], fitness[i] = self.simulated_annealing(population[i], trial, fitness[i], f_trial)
                if fitness[i] < self.f_opt:
                    self.f_opt = fitness[i]
                    self.x_opt = population[i]
            self.temp *= self.cooling_rate
        
        return self.f_opt, self.x_opt
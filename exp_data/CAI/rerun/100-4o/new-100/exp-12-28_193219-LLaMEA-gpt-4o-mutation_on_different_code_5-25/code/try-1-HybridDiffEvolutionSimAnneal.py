import numpy as np

class HybridDiffEvolutionSimAnneal:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 50
        self.F = 0.8  # Differential evolution scaling factor
        self.CR = 0.9  # Crossover probability
        self.temp = 100  # Initial temperature for simulated annealing

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for generation in range(self.budget // self.population_size):
            dynamic_CR = self.CR - (0.5 * generation / (self.budget // self.population_size))
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, bounds[0], bounds[1])
                crossover = np.random.rand(self.dim) < dynamic_CR
                trial = np.where(crossover, mutant, population[i])
                
                f_trial = func(trial)
                
                if f_trial < fitness[i] or np.exp((fitness[i] - f_trial) / self.temp) > np.random.rand():
                    population[i] = trial
                    fitness[i] = f_trial

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
            
            self.temp *= 0.99  # Simulated annealing temperature decay

        return self.f_opt, self.x_opt
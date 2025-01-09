import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=None):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size or 10 * dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.mutation_factor = 0.5
        self.crossover_prob = 0.5

    def __call__(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        num_evaluations = self.population_size
        
        while num_evaluations < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                
                mutant = np.clip(a + self.mutation_factor * (b - c), self.bounds[0], self.bounds[1])
                crossover = np.random.rand(self.dim) < self.crossover_prob
                trial = np.where(crossover, mutant, self.population[i])
                
                f_trial = func(trial)
                num_evaluations += 1
                
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    self.population[i] = trial
                    
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                
                if num_evaluations >= self.budget:
                    break

            self.adaptive_parameters(fitness)

        return self.f_opt, self.x_opt

    def adaptive_parameters(self, fitness):
        # Simple adaptive scheme to adjust mutation factor and crossover probability
        self.mutation_factor = 0.4 + 0.1 * np.random.rand()
        self.crossover_prob = 0.3 + 0.4 * np.random.rand()
        # Adjust population size dynamically based on convergence
        self.population_size = max(5, int(self.population_size * (1 + 0.1 * np.random.randn())))
        self.population = self.population[:self.population_size]
        fitness.resize(self.population_size)
import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size
        generation = 0

        while evaluations < self.budget:
            new_pop_size = max(10, int(self.pop_size * (1 - generation / (self.budget // self.pop_size))))
            new_population = np.empty((new_pop_size, self.dim))
            new_fitness = np.empty(new_pop_size)

            for i in range(new_pop_size):
                a, b, c = np.random.choice([x for x in range(self.pop_size) if x != i], 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lb, self.ub)
                
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i % self.pop_size])
                
                trial_fitness = func(trial)
                evaluations += 1
                
                if trial_fitness < fitness[i % self.pop_size]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial
                else:
                    new_population[i] = population[i % self.pop_size]
                    new_fitness[i] = fitness[i % self.pop_size]

                if evaluations >= self.budget:
                    break
                
            population = new_population
            fitness = new_fitness
            self.pop_size = new_pop_size
            generation += 1
        
        return self.f_opt, self.x_opt
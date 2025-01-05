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
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        # Track the number of evaluations
        evaluations = self.pop_size

        # DE loop
        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Mutation & Crossover (DE/rand/1/bin)
                a, b, c = np.random.choice([x for x in range(self.pop_size) if x != i], 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lb, self.ub)
                
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    # Update the best solution found
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

                if evaluations >= self.budget:
                    break
        
        return self.f_opt, self.x_opt
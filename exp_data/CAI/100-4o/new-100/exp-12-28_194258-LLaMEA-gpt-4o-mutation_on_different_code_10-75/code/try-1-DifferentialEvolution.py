import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Mutation factor
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub
        population = np.random.uniform(lower_bound, upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size
        
        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Mutation
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                F_dynamic = self.F + np.random.uniform(-0.1, 0.1)  # Adaptive F
                mutant = population[a] + F_dynamic * (population[b] - population[c])
                mutant = np.clip(mutant, lower_bound, upper_bound)

                # Crossover
                trial = np.copy(population[i])
                CR_dynamic = self.CR + np.random.uniform(-0.1, 0.1)  # Adaptive CR
                crossover_mask = np.random.rand(self.dim) < CR_dynamic
                trial[crossover_mask] = mutant[crossover_mask]

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial
                
                if evaluations >= self.budget:
                    break
            
        return self.f_opt, self.x_opt
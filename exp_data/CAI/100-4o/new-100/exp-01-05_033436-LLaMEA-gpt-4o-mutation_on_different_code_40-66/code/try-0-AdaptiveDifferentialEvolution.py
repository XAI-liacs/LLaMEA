import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)  # Adaptive population size
        self.f_opt = np.Inf
        self.x_opt = None
        self.F = 0.5  # Scaling factor
        self.CR = 0.9  # Crossover rate

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        pop = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evals = self.population_size
        
        while evals < self.budget:
            for i in range(self.population_size):
                if evals >= self.budget:
                    break

                # Mutation
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds[0], bounds[1])

                # Crossover
                trial = np.copy(pop[i])
                crossover_points = np.random.rand(self.dim) < self.CR
                trial[crossover_points] = mutant[crossover_points]

                # Selection
                trial_fitness = func(trial)
                evals += 1
                
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                
                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial
            
            # Adaptive control parameters
            self.F = np.random.uniform(0.4, 0.9)
            self.CR = np.random.uniform(0.7, 1.0)

        return self.f_opt, self.x_opt
import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Differential Evolution Parameters
        F = 0.5  # Initial differential weight
        Cr = 0.7  # Initial crossover probability
        bounds = (func.bounds.lb, func.bounds.ub)
        
        # Initialize population
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        # Update best known solution
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        
        evaluations = self.pop_size
        
        # Main loop
        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Mutation (DE/rand/1)
                indices = np.random.permutation(self.pop_size)
                indices = indices[indices != i][:3]
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + F * (x2 - x3), bounds[0], bounds[1])
                
                # Crossover
                cross_points = np.random.rand(self.dim) < Cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                
                # Update best solution
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
            
            # Adapt F and Cr based on success rates
            if evaluations % self.pop_size == 0:
                success_rate = np.mean(fitness < self.f_opt)
                F = np.clip(F + 0.1 * (success_rate - 0.2), 0.1, 1)
                Cr = np.clip(Cr + 0.1 * (0.9 - success_rate), 0.1, 1)
                # Dynamic population scaling
                convergence = np.std(fitness) / np.abs(np.mean(fitness))
                self.pop_size = max(20, int(self.pop_size * (1 - 0.2 * convergence)))
        
        return self.f_opt, self.x_opt
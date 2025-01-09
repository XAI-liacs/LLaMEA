import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None
    
    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        
        # Initialize population
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        # Store best solution
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx].copy()
        
        # Setting initial parameters
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        eval_count = self.population_size
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                # Mutation: select three random indices different from i
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                # Adaptive mutation factor based on diversity
                F_adaptive = 0.5 + (np.std(fitness) / np.mean(fitness))
                F_adaptive = np.clip(F_adaptive, 0.5, 1.0)
                
                # Create donor vector
                mutant = np.clip(a + F_adaptive * (b - c), bounds[0], bounds[1])
                
                # Dynamic crossover rate based on individual's performance
                CR_adaptive = CR * (fitness[i] / self.f_opt)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < CR_adaptive
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                # Evaluate trial vector
                f_trial = func(trial)
                eval_count += 1
                
                # Selection
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    
                    # Update best solution if found
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()
                
                # Stop if budget exhausted
                if eval_count >= self.budget:
                    break
        
        return self.f_opt, self.x_opt
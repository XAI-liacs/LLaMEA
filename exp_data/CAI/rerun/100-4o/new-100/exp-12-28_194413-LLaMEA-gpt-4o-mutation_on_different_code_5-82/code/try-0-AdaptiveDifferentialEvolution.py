import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 20  # Population size
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        bounds = [func.bounds.lb, func.bounds.ub]
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx].copy()
        
        evaluations = self.pop_size
        
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                    
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                mutant = np.clip(a + self.F * (b - c), bounds[0], bounds[1])
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                f_trial = func(trial)
                evaluations += 1
                
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()
                        
            # Adapt F and CR based on success rate
            if evaluations % 100 == 0:  # Adjust parameters every 100 evaluations
                success_rate = np.sum(fitness < self.f_opt) / self.pop_size
                self.F = 0.1 + 0.4 * success_rate  # Adaptive F
                self.CR = 0.5 + 0.4 * success_rate  # Adaptive CR

        return self.f_opt, self.x_opt
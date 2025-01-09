import numpy as np

class ImprovedAdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=20):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = None
        self.bounds = (-5.0, 5.0)
        self.success_history = []
        self.F = 0.5
        self.CR = 0.7

    def __call__(self, func):
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], 
                                            (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in self.population])
        
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = self.population[best_idx]
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[indices]
                
                self.F = np.mean(self.success_history[-10:]) if len(self.success_history) >= 10 else np.random.uniform(0.5, 1.0)
                mutant = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])
                
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    self.population[i] = trial
                    fitness[i] = f_trial
                    self.success_history.append(self.F)
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

            self.population_size = int(self.population_size * (0.9 + 0.1 * (1 - evaluations / self.budget)))
            self.CR = np.random.uniform(0.1, 0.9)

        return self.f_opt, self.x_opt
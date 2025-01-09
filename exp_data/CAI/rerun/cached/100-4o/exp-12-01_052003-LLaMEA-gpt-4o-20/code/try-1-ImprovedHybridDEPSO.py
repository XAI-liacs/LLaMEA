import numpy as np

class ImprovedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 20
        self.F = 0.5  # DE scaling factor
        self.CR = 0.9 # Crossover rate
        self.w = 0.5  # Inertia weight for PSO
        self.c1 = 1.5 # Cognitive component for PSO
        self.c2 = 1.5 # Social component for PSO
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.best_individual = None
        self.best_fitness = np.inf
        self.personal_best = self.population.copy()
        self.personal_best_fitness = np.full(self.pop_size, np.inf)
        
    def __call__(self, func):
        evaluations = 0
        
        # Evaluate initial population
        for i in range(self.pop_size):
            fitness = func(self.population[i])
            evaluations += 1
            if fitness < self.personal_best_fitness[i]:
                self.personal_best[i] = self.population[i].copy()
                self.personal_best_fitness[i] = fitness
            if fitness < self.best_fitness:
                self.best_individual = self.population[i].copy()
                self.best_fitness = fitness
                
        # Optimization loop
        while evaluations < self.budget:
            # Differential Evolution and PSO hybrid step
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                # Adaptive DE mutation
                self.F = 0.5 + 0.5 * (self.best_fitness / max(self.personal_best_fitness))
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                
                # DE crossover with tournament selection
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])
                
                # Evaluate trial vector
                trial_fitness = func(trial)
                evaluations += 1
                
                # Selection
                if trial_fitness < self.personal_best_fitness[i]:
                    self.personal_best[i] = trial
                    self.personal_best_fitness[i] = trial_fitness
                    if trial_fitness < self.best_fitness:
                        self.best_individual = trial
                        self.best_fitness = trial_fitness
                
                # PSO velocity update with adaptive inertia
                self.w = 0.5 + 0.5 * (self.best_fitness / max(self.personal_best_fitness))
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocities[i] = (self.w * self.velocities[i] 
                                      + self.c1 * r1 * (self.personal_best[i] - self.population[i])
                                      + self.c2 * r2 * (self.best_individual - self.population[i]))
                
                # PSO position update
                self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lower_bound, self.upper_bound)
                
        return self.best_individual, self.best_fitness
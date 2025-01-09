import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.inf
        self.x_opt = None
        
        # Initialize population
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        
        # Adaptive parameters
        self.F = 0.5  # Differential weight
        self.CR = 0.9 # Crossover probability

    def __call__(self, func):
        evaluations = 0
        
        # Evaluate initial population
        for i in range(self.pop_size):
            self.fitness[i] = func(self.population[i])
            evaluations += 1
            if self.fitness[i] < self.f_opt:
                self.f_opt = self.fitness[i]
                self.x_opt = self.population[i].copy()
                
        # Main loop
        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Select three indices different from i
                indices = np.random.choice(self.pop_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = self.population[indices]
                
                # Mutation and crossover
                mutant = np.clip(a + self.F * (b - c), -5.0, 5.0)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.population[i])
                
                # Evaluate trial vector
                f_trial = func(trial)
                evaluations += 1
                
                # Selection
                if f_trial < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()
                        
                # Local search around the best individual
                if evaluations % (self.pop_size // 2) == 0:
                    local_trial = np.clip(self.x_opt + np.random.normal(0, 0.1, self.dim), -5.0, 5.0)
                    f_local_trial = func(local_trial)
                    evaluations += 1
                    if f_local_trial < self.f_opt:
                        self.f_opt = f_local_trial
                        self.x_opt = local_trial

                # Adapt parameters F and CR
                self.F = np.abs(np.random.normal(0.5, 0.3))
                self.CR = np.clip(np.random.normal(0.9, 0.1), 0.1, 1.0)
                
                if evaluations >= self.budget:
                    break
                
        return self.f_opt, self.x_opt
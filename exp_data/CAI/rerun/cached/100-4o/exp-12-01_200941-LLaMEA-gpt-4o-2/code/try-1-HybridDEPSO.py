import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20 + 5 * int(np.sqrt(dim))  # Adjusted for smaller populations in higher dimensions
        self.f = 0.8  # Differential weight
        self.cr = 0.9  # Crossover probability
        self.w = 0.9  # Inertia weight for PSO
        self.c1 = 1.5 # Cognitive component for PSO
        self.c2 = 1.5 # Social component for PSO

    def __call__(self, func):
        np.random.seed(42)
        evaluations = 0
        
        # Initialize population for DE
        de_population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        de_fitness = np.apply_along_axis(func, 1, de_population)
        evaluations += self.population_size
        
        # Initialize velocity and personal bests for PSO
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = np.copy(de_population)
        personal_best_fitness = np.copy(de_fitness)
        global_best_idx = np.argmin(de_fitness)
        global_best = de_population[global_best_idx]
        global_best_fitness = de_fitness[global_best_idx]
        
        # Hybrid DE-PSO loop
        while evaluations < self.budget:
            # Differential Evolution Mutation and Crossover
            for i in range(self.population_size):
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                a, b, c = de_population[indices]
                mutant = np.clip(a + self.f * (b - c), self.lower_bound, self.upper_bound)
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, de_population[i])
                
                # Evaluate trial
                trial_fitness = func(trial)
                evaluations += 1
                
                if trial_fitness < de_fitness[i]:
                    de_population[i] = trial
                    de_fitness[i] = trial_fitness
                    if trial_fitness < personal_best_fitness[i]:
                        personal_best[i] = trial
                        personal_best_fitness[i] = trial_fitness
                        if trial_fitness < global_best_fitness:
                            global_best = trial
                            global_best_fitness = trial_fitness
            
            # Particle Swarm Optimization Update
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best[i] - de_population[i]) +
                               self.c2 * r2 * (global_best - de_population[i]))
                de_population[i] = np.clip(de_population[i] + velocity[i], self.lower_bound, self.upper_bound)
                
                # Evaluate new position
                new_fitness = func(de_population[i])
                evaluations += 1
                
                if new_fitness < personal_best_fitness[i]:
                    personal_best[i] = de_population[i]
                    personal_best_fitness[i] = new_fitness
                    if new_fitness < global_best_fitness:
                        global_best = de_population[i]
                        global_best_fitness = new_fitness

        return global_best, global_best_fitness
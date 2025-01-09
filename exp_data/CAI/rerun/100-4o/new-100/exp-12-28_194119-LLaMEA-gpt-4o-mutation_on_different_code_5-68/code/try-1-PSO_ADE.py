import numpy as np

class PSO_ADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 40
        self.pop = np.random.uniform(-5.0, 5.0, (self.population_size, dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.population_size, dim))
        self.personal_best = self.pop.copy()
        self.personal_best_fitness = np.full(self.population_size, np.Inf)
        self.global_best = None
        self.global_best_fitness = np.Inf
        self.F = 0.5  # DE mutation factor
        self.CR = 0.9  # DE crossover probability
        
    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            success_count = 0  # Track successful improvements
            for i in range(self.population_size):
                fitness = func(self.pop[i])
                evaluations += 1
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.pop[i].copy()
                    success_count += 1  # Count successful updates
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.pop[i].copy()
            
            # PSO Update
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            self.velocities = 0.7 * self.velocities + 1.5 * r1 * (self.personal_best - self.pop) + 1.5 * r2 * (self.global_best - self.pop)
            self.pop += self.velocities
            self.pop = np.clip(self.pop, -5.0, 5.0)
            
            # DE Update
            adaptive_F = self.F * (1 + 0.1 * success_count / self.population_size)  # Adjust F based on success rate
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = self.pop[a] + adaptive_F * (self.pop[b] - self.pop[c])  # Use adaptive F
                mutant_vector = np.clip(mutant_vector, -5.0, 5.0)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(cross_points, mutant_vector, self.pop[i])
                
                trial_fitness = func(trial_vector)
                evaluations += 1
                if trial_fitness < self.personal_best_fitness[i]:
                    self.pop[i] = trial_vector
                    self.personal_best_fitness[i] = trial_fitness
                    self.personal_best[i] = trial_vector.copy()
                if trial_fitness < self.global_best_fitness:
                    self.global_best_fitness = trial_fitness
                    self.global_best = trial_vector.copy()
                
                if evaluations >= self.budget:
                    break
        
        self.f_opt = self.global_best_fitness
        self.x_opt = self.global_best
        return self.f_opt, self.x_opt
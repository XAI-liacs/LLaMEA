import numpy as np

class EnhancedHybridAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.init_pop_size = 6 * dim  # Reduced initial population size
        self.min_pop_size = 4 * dim  # Minimum population size to maintain diversity
        self.max_pop_size = 10 * dim  # Maximum population size to adapt search radius
        self.current_pop_size = self.init_pop_size
        self.F = np.random.uniform(0.4, 0.9)
        self.CR = np.random.uniform(0.5, 0.9)
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.current_pop_size, self.dim))
        self.fitness = np.full(self.current_pop_size, np.inf)
        self.elite_ratio = 0.3  # Increased elite ratio
        self.elite_count = max(1, int(self.current_pop_size * self.elite_ratio))
        self.sigma = np.random.uniform(0.2, 0.4)

    def mutate(self, idx):
        elite_indices = np.argsort(self.fitness)[:self.elite_count]
        idxs = np.random.choice(elite_indices, 2, replace=False)
        a, b = self.population[idxs]
        c = self.population[np.random.choice(np.delete(np.arange(self.current_pop_size), np.concatenate(([idx], idxs))))]
        return a + self.F * (b - c)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def adapt_parameters(self, trial, target_idx, func):
        trial_fitness = func(trial)
        if trial_fitness < self.fitness[target_idx]:
            self.population[target_idx] = trial
            self.fitness[target_idx] = trial_fitness
            return True, trial_fitness
        return False, self.fitness[target_idx]

    def covariance_mutation(self, idx):
        mean = np.mean(self.population, axis=0)
        cov = np.cov(self.population, rowvar=False)
        return np.random.multivariate_normal(mean, self.sigma * cov)

    def __call__(self, func):
        evaluations = 0
        self.fitness = np.array([func(ind) for ind in self.population])
        evaluations += self.current_pop_size
        
        while evaluations < self.budget:
            for i in range(self.current_pop_size):
                if evaluations >= self.budget:
                    break
                
                if np.random.rand() < 0.5:
                    mutant_vec = self.mutate(i)
                else:
                    mutant_vec = self.covariance_mutation(i)
                
                mutant_vec = np.clip(mutant_vec, self.lower_bound, self.upper_bound)
                trial_vec = self.crossover(self.population[i], mutant_vec)
                
                successful, new_fitness = self.adapt_parameters(trial_vec, i, func)
                evaluations += 1

            success_rate = np.sum(self.fitness < np.median(self.fitness)) / self.current_pop_size
            self.F = np.clip(self.F * (1 + (0.3 if success_rate > 0.3 else -0.2)), 0.1, 1.0)
            self.CR = np.clip(self.CR * (1 + (0.3 if success_rate > 0.3 else -0.2)), 0.1, 1.0)
            self.sigma = np.clip(self.sigma * (1 + (0.2 if success_rate > 0.3 else -0.1)), 0.1, 1.0)
            # Adjust population size dynamically
            if success_rate > 0.5:
                self.current_pop_size = min(self.current_pop_size + 2, self.max_pop_size)
            else:
                self.current_pop_size = max(self.current_pop_size - 2, self.min_pop_size)
            self.population = self.population[:self.current_pop_size]
            self.fitness = self.fitness[:self.current_pop_size]

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]
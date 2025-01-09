import numpy as np

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(max(4, dim * 10), budget // 10)
        self.F = 0.8  # Scaling factor for mutation
        self.CR = 0.9 # Crossover probability

    def __call__(self, func):
        np.random.seed(0)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Step 1: Differential Evolution - Mutation and Crossover
                indices = list(range(self.population_size))
                indices.remove(i)
                np.random.shuffle(indices)
                a, b, c = population[indices[:3]]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                # Step 2: Evaluate trial vector
                trial_fitness = func(trial)
                evaluations += 1

                # Step 3: Selection
                if trial_fitness < fitness[i]: 
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Step 4: Local Search
                if evaluations < self.budget:
                    local_point = population[i] + np.random.normal(0, 0.1, self.dim)
                    local_point = np.clip(local_point, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_point)
                    evaluations += 1
                    if local_fitness < fitness[i]:
                        population[i] = local_point
                        fitness[i] = local_fitness

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.8  # Differential weight
        self.CR = 0.9 # Crossover probability
    
    def __call__(self, func):
        np.random.seed(42)
        
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, 
                                       (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Randomly select three distinct vectors
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                # Mutation
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Evaluate trial vector
                f_trial = func(trial)
                evaluations += 1
                
                # Selection
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                
                # Adaptive Local Search
                if evaluations < self.budget and np.random.rand() < 0.1:
                    local_search_vector = trial + np.random.normal(0, 0.1, self.dim)
                    local_search_vector = np.clip(local_search_vector, self.lower_bound, self.upper_bound)
                    f_local = func(local_search_vector)
                    evaluations += 1
                    if f_local < f_trial:
                        population[i] = local_search_vector
                        fitness[i] = f_local

                if evaluations >= self.budget:
                    break
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
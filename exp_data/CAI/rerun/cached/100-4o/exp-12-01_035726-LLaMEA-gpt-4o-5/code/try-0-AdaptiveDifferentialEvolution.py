import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # a common choice for DE
        self.bounds = (-5.0, 5.0)
    
    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], 
                                       (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        
        evaluations = self.population_size
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                # Mutation
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = population[np.random.choice(candidates, 3, replace=False)]
                F = np.random.uniform(0.5, 1.0)  # dynamic scaling factor
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])
                
                # Crossover
                cross_points = np.random.rand(self.dim) < np.random.uniform(0.1, 0.9)
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best = trial
                        
        return best
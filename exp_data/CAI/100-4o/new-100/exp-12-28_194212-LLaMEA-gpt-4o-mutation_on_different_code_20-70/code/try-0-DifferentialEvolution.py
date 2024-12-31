import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50, crossover_prob=0.9, differential_weight=0.8):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.differential_weight = differential_weight
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        # Initial population is randomly generated within the bounds
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.differential_weight * (population[b] - population[c])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial
                
                # Update the best solution found
                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial
                
                if evaluations >= self.budget:
                    break
        
        return self.f_opt, self.x_opt
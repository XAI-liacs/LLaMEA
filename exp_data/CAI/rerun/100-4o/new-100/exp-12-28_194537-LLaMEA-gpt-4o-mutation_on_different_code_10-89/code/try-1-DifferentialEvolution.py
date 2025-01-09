import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=20, mutation_factor=0.8, crossover_prob=0.7):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.f_opt, best_idx = np.min(fitness), np.argmin(fitness)
        self.x_opt = population[best_idx].copy()

        evaluations = self.population_size

        while evaluations < self.budget:
            if evaluations % (self.budget // 5) == 0:  # Periodic reinitialization
                population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
                fitness = np.array([func(ind) for ind in population])
            
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                adaptive_factor = np.random.uniform(0.5, 1.0)  # Adaptive mutation factor
                mutant = population[a] + adaptive_factor * (population[b] - population[c])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                f_trial = func(trial)
                evaluations += 1
                
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                
                if evaluations >= self.budget:
                    break
        
        return self.f_opt, self.x_opt
import numpy as np

class EHQEA_Swarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.5
        self.elite_fraction = 0.1
        self.inertia_weight = 0.7
        self.cognitive_param = 1.5
        self.social_param = 2.0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        personal_best = population.copy()
        personal_best_fitness = fitness.copy()
        best_idx = np.argmin(fitness)
        global_best = population[best_idx]
        global_best_fitness = fitness[best_idx]
        evaluations = self.population_size

        while evaluations < self.budget:
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocity = (self.inertia_weight * velocity +
                        self.cognitive_param * r1 * (personal_best - population) +
                        self.social_param * r2 * (global_best - population))
            population += velocity
            population = np.clip(population, lb, ub)
            
            fitness = np.array([func(ind) for ind in population])
            evaluations += self.population_size
            
            improved = fitness < personal_best_fitness
            personal_best[improved] = population[improved]
            personal_best_fitness[improved] = fitness[improved]
            
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < global_best_fitness:
                global_best, global_best_fitness = population[current_best_idx], fitness[current_best_idx]
        
        return global_best, global_best_fitness
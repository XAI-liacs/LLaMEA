import numpy as np

class AOPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.inertia_weight = 0.9
        self.inertia_damping = 0.99
        self.c1 = 2.0
        self.c2 = 2.0
    
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        velocity_bounds = 0.2 * (bounds[1] - bounds[0])
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        velocity = np.random.uniform(-velocity_bounds, velocity_bounds, (self.population_size, self.dim))
        personal_best = population.copy()
        personal_best_fitness = np.array([func(ind) for ind in personal_best])
        global_best_index = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_index]
        global_best_fitness = personal_best_fitness[global_best_index]
        
        eval_count = self.population_size
        
        while eval_count < self.budget:
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocity = (self.inertia_weight * velocity
                        + self.c1 * r1 * (personal_best - population)
                        + self.c2 * r2 * (global_best - population))
            population += velocity
            population = np.clip(population, bounds[0], bounds[1])
            
            fitness = np.array([func(ind) for ind in population])
            eval_count += self.population_size
            
            # Update personal best
            better_mask = fitness < personal_best_fitness
            personal_best[better_mask] = population[better_mask]
            personal_best_fitness[better_mask] = fitness[better_mask]
            
            # Update global best
            current_global_best_index = np.argmin(personal_best_fitness)
            if personal_best_fitness[current_global_best_index] < global_best_fitness:
                global_best = personal_best[current_global_best_index]
                global_best_fitness = personal_best_fitness[current_global_best_index]
            
            # Introduce Quasi-Oppositional Learning
            opposite_population = bounds[0] + bounds[1] - population
            opposite_fitness = np.array([func(ind) for ind in opposite_population])
            eval_count += self.population_size
            better_opposite_mask = opposite_fitness < fitness
            population[better_opposite_mask] = opposite_population[better_opposite_mask]
            fitness[better_opposite_mask] = opposite_fitness[better_opposite_mask]
            
            # Update inertia weight
            self.inertia_weight *= self.inertia_damping
        
        return global_best
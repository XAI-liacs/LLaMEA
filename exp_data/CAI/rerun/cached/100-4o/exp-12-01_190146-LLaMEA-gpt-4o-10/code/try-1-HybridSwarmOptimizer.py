import numpy as np

class HybridSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.w = 0.5   # Inertia weight
        self.F = 0.8   # Differential weight
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        # Initialize the population and velocities for PSO
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        
        # Evaluate initial population
        fitness = np.apply_along_axis(func, 1, population)
        budget_used = self.population_size
        
        # Initialize personal bests and global best
        p_best_positions = np.copy(population)
        p_best_fitness = np.copy(fitness)
        g_best_position = p_best_positions[np.argmin(p_best_fitness)]
        g_best_fitness = np.min(p_best_fitness)
        
        while budget_used < self.budget:
            # Update velocities and positions for PSO
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            self.w = 0.9 - (0.5 * budget_used / self.budget)  # Adaptive inertia weight
            velocities = (self.w * velocities +
                          self.c1 * r1 * (p_best_positions - population) +
                          self.c2 * r2 * (g_best_position - population))
            population = np.clip(population + velocities, self.lower_bound, self.upper_bound)

            # Differential Evolution Mutation and Crossover
            for i in range(self.population_size):
                if budget_used >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                crossover = np.random.rand(self.dim) < self.CR
                candidate = np.where(crossover, mutant, population[i])
                
                # Evaluate candidate solution
                candidate_fitness = func(candidate)
                budget_used += 1

                # Select the better solution
                if candidate_fitness < fitness[i]:
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    if candidate_fitness < p_best_fitness[i]:
                        p_best_positions[i] = candidate
                        p_best_fitness[i] = candidate_fitness
            
            # Update global best
            current_g_best_index = np.argmin(p_best_fitness)
            if p_best_fitness[current_g_best_index] < g_best_fitness:
                g_best_position = p_best_positions[current_g_best_index]
                g_best_fitness = p_best_fitness[current_g_best_index]

        return g_best_position, g_best_fitness
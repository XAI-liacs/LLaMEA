import numpy as np

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.f = 0.8  # Differential weight
        self.cr = 0.9  # Crossover probability
        self.current_eval = 0

    def __call__(self, func):
        # Initialize the population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        self.current_eval += self.population_size
        
        # Main loop
        while self.current_eval < self.budget:
            for i in range(self.population_size):
                # Mutation
                idxs = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[idxs]
                self.f = 0.5 + 0.3 * np.random.rand()  # Adaptive differential weight
                mutant = np.clip(x0 + self.f * (x1 - x2), self.lower_bound, self.upper_bound)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                self.current_eval += 1
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial

                # Early stopping if budget is exhausted
                if self.current_eval >= self.budget:
                    break

            # Adaptive Local Search
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]
            if np.random.rand() < 0.2:  # 20% chance to perform local search
                perturbation = np.random.normal(0, 0.1, self.dim)
                local_solution = np.clip(best_individual + perturbation, self.lower_bound, self.upper_bound)
                local_fitness = func(local_solution)
                self.current_eval += 1
                if local_fitness < fitness[best_idx]:
                    fitness[best_idx] = local_fitness
                    population[best_idx] = local_solution

        return population[np.argmin(fitness)]

# Example usage:
# optimizer = HybridDEOptimizer(budget=10000, dim=10)
# best_solution = optimizer(func)  # where func is the black box function to optimize
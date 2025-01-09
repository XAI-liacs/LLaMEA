import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + int(2 * np.sqrt(self.dim))
        self.scale_factor = 0.8
        self.crossover_rate = 0.9
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def __call__(self, func):
        # Initialize population
        population = self.lower_bound + (self.upper_bound - self.lower_bound) * np.random.rand(self.population_size, self.dim)
        fitness = np.apply_along_axis(func, 1, population)
        eval_count = self.population_size
        
        while eval_count < self.budget:
            # Differential Evolution Step
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                # Adaptive scale factor
                adaptive_scale_factor = self.scale_factor * np.random.uniform(0.5, 1.5)
                mutant = np.clip(a + adaptive_scale_factor * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                eval_count += 1

                # Greedy Selection
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial

                    # Update best solution
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial

                if eval_count >= self.budget:
                    break
            
            # Adaptive Local Search
            if eval_count < self.budget:
                for i in range(self.population_size):
                    if eval_count >= self.budget:
                        break
                    # Simple local search move
                    candidate = population[i] + np.random.normal(0, 0.1, self.dim)
                    candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                    candidate_fitness = func(candidate)
                    eval_count += 1

                    if candidate_fitness < fitness[i]:
                        fitness[i] = candidate_fitness
                        population[i] = candidate

                        if candidate_fitness < self.best_fitness:
                            self.best_fitness = candidate_fitness
                            self.best_solution = candidate

        return self.best_solution
import numpy as np

class HybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.temperature = 100.0  # Initial temperature for SA

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation (DE)
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    # Simulated Annealing acceptance
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness
                    else:
                        if np.random.rand() < np.exp((best_fitness - trial_fitness) / self.temperature):
                            best_solution = trial
                            best_fitness = trial_fitness
            
            # Annealing schedule
            self.temperature *= 0.99

        return best_solution

# Example usage:
# optimizer = HybridDE_SA(budget=1000, dim=5)
# func = lambda x: np.sum(x**2)
# best_solution = optimizer(func)
# print("Best solution found:", best_solution)
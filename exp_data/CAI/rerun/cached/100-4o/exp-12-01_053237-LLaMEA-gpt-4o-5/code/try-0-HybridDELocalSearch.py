import numpy as np

class HybridDELocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(5 * dim, budget // 2)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.best_solution = None
        self.best_value = float('inf')

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= self.population_size
        
        while self.budget > 0:
            for i in range(self.population_size):
                # Mutation and Crossover
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                # Local Search: Small perturbation
                if np.random.rand() < 0.2:
                    trial = self.local_search(trial, func)
                
                # Selection
                trial_fitness = func(trial)
                self.budget -= 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < self.best_value:
                        self.best_value = trial_fitness
                        self.best_solution = trial
                
                if self.budget <= 0:
                    break

        return self.best_solution

    def local_search(self, solution, func):
        perturbation = np.random.normal(0, 0.1, size=self.dim)
        perturbed_solution = np.clip(solution + perturbation, self.lower_bound, self.upper_bound)
        if func(perturbed_solution) < func(solution):
            return perturbed_solution
        return solution
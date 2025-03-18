import numpy as np
from scipy.optimize import minimize

class HybridDE:
    def __init__(self, budget, dim, population_size=50, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.evaluations = 0

    def quasi_oppositional_initialization(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        opposition = lb + ub - population
        combined = np.vstack((population, opposition))
        return combined[np.random.choice(2 * self.population_size, self.population_size, replace=False)]

    def differential_evolution_step(self, population, bounds, func):
        new_population = np.empty_like(population)
        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + self.F * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < (self.CR * (1.1 if np.random.rand() > 0.5 else 0.9))  # Dynamic crossover rate
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            trial_eval = func(trial)
            self.evaluations += 1
            if trial_eval < func(population[i]):
                new_population[i] = trial
            else:
                new_population[i] = population[i]
        return new_population

    def local_refinement(self, best_solution, bounds, func):
        result = minimize(func, best_solution, method='L-BFGS-B', bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)])
        return result.x

    def __call__(self, func):
        bounds = func.bounds
        
        # Initial quasi-oppositional population
        population = self.quasi_oppositional_initialization(bounds)
        
        # Evaluate initial population
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += self.population_size
        
        while self.evaluations < self.budget:
            # Perform a DE step
            population = self.differential_evolution_step(population, bounds, func)
            
            # Update fitness based on new population
            fitness = np.array([func(ind) for ind in population])
            self.evaluations += self.population_size
            
            # Check if budget is exceeded
            if self.evaluations >= self.budget:
                break

        # Find the best solution from the population
        best_index = np.argmin(fitness)
        best_solution = population[best_index]

        # Local refinement using BFGS
        refined_solution = self.local_refinement(best_solution, bounds, func)

        return refined_solution
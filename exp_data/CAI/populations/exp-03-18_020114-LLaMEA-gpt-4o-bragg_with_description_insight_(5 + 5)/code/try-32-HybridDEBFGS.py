import numpy as np
from scipy.optimize import minimize

class HybridDEBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.current_budget = 0
        self.population_size = 20
        self.pop = None
        self.bounds = None
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.period = self.dim // 2

    def initialize_population(self, lb, ub):
        self.pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        
    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.pop])
        return fitness
    
    def differential_evolution(self, func):
        fitness = self.evaluate_population(func)
        while self.current_budget < self.budget:
            for i in range(self.population_size):
                donors = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.pop[donors]
                self.mutation_factor = 0.5 + 0.4 * np.cos(np.pi * self.current_budget / self.budget)  # Annealed mutation factor
                mutant = np.clip(x0 + self.mutation_factor * (x1 - x2), self.bounds.lb, self.bounds.ub)
                trial = np.copy(self.pop[i])
                crossover = np.random.rand(self.dim) < self.crossover_prob
                trial[crossover] = mutant[crossover]
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    self.pop[i] = trial
                    fitness[i] = trial_fitness
                self.current_budget += 1
                if self.current_budget >= self.budget:
                    break
        return self.pop[np.argmin(fitness)]
    
    def local_optimization(self, func, solution):
        result = minimize(func, solution, method='L-BFGS-B', bounds=list(zip(self.bounds.lb, self.bounds.ub)))
        return result.x
    
    def encourage_periodicity(self, solution):
        for i in range(self.dim):
            solution[i] = (solution[i] + solution[(i + self.period) % self.dim]) / 2
        return solution

    def symmetry_aware_optimization(self, solution):
        for i in range(self.period):
            solution[i] = solution[self.dim - i - 1]  # Symmetrize layers
        return solution

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population(self.bounds.lb, self.bounds.ub)

        # Differential Evolution for global search
        best_solution = self.differential_evolution(func)

        # Local optimization for fine-tuning
        best_solution = self.local_optimization(func, best_solution)
        
        # Encourage periodicity
        best_solution = self.encourage_periodicity(best_solution)

        # Symmetry-aware optimization
        best_solution = self.symmetry_aware_optimization(best_solution)
        
        return best_solution
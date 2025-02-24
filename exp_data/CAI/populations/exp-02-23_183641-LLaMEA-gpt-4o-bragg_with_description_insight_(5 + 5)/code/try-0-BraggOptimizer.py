import numpy as np
from scipy.optimize import minimize

class BraggOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(20, budget // dim)
        self.pop = None
        self.best_solution = None
        self.best_value = float('-inf')
        self.bounds = None
        
    def initialize_population(self, bounds):
        lb, ub = bounds
        self.pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        # Quasi-oppositional initialization
        opposite_pop = lb + ub - self.pop
        self.pop = np.concatenate((self.pop, opposite_pop))
        
    def differential_evolution_step(self, func):
        F = 0.8  # Differential weight
        CR = 0.9 # Crossover probability
        new_pop = np.copy(self.pop)
        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            R = np.random.randint(self.dim)
            trial = np.copy(self.pop[i])
            for j in range(self.dim):
                if np.random.rand() < CR or j == R:
                    trial[j] = self.pop[a][j] + F * (self.pop[b][j] - self.pop[c][j])
                    # Enforce bounds
                    trial[j] = np.clip(trial[j], self.bounds[0][j], self.bounds[1][j])
            # Evaluate trial solution
            trial_value = func(trial)
            if trial_value > func(self.pop[i]):
                new_pop[i] = trial
                if trial_value > self.best_value:
                    self.best_value = trial_value
                    self.best_solution = trial
        self.pop = new_pop
        
    def local_optimization(self, func):
        if self.best_solution is not None:
            result = minimize(lambda x: -func(x), self.best_solution, bounds=self.bounds, method='L-BFGS-B')
            if -result.fun > self.best_value:
                self.best_value = -result.fun
                self.best_solution = result.x

    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        # Initialize population
        self.initialize_population(self.bounds)
        evaluations = 0
        
        # Main optimization loop
        while evaluations < self.budget:
            self.differential_evolution_step(func)
            evaluations += self.population_size
            if evaluations + self.dim <= self.budget:
                # Local optimization
                self.local_optimization(func)
                evaluations += self.dim

        return self.best_solution
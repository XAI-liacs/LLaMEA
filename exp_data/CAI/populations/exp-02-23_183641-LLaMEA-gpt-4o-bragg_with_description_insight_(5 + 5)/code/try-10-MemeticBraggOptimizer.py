import numpy as np
from scipy.optimize import minimize

class MemeticBraggOptimizer:
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
        opposite_pop = lb + ub - self.pop
        self.pop = np.concatenate((self.pop, opposite_pop))
        
    def differential_evolution_cc(self, func):
        F = 0.8 
        CR = 0.9 
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
                    trial[j] = np.clip(trial[j], self.bounds[0][j], self.bounds[1][j])
            trial_value = func(trial)
            if trial_value > func(self.pop[i]):
                new_pop[i] = trial
                if trial_value > self.best_value:
                    self.best_value = trial_value
                    self.best_solution = trial
        self.pop = new_pop
        
    def simulated_annealing_step(self, func, temp):
        if self.best_solution is not None:
            candidate = self.best_solution + np.random.uniform(-0.05, 0.05, self.dim)
            candidate = np.clip(candidate, self.bounds[0], self.bounds[1])
            candidate_value = func(candidate)
            if candidate_value > self.best_value or np.exp((candidate_value - self.best_value) / temp) > np.random.rand():
                self.best_value = candidate_value
                self.best_solution = candidate
                
    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        self.initialize_population(self.bounds)
        evaluations = 0
        temperature = 1.0
        cooling_rate = 0.95
        
        while evaluations < self.budget:
            self.differential_evolution_cc(func)
            evaluations += self.population_size
            temperature *= cooling_rate
            
            if evaluations + self.dim <= self.budget:
                self.simulated_annealing_step(func, temperature)
                evaluations += self.dim

        return self.best_solution
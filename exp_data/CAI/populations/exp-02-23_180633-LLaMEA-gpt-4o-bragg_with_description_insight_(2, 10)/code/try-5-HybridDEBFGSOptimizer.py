import numpy as np
from scipy.optimize import minimize

class HybridDEBFGSOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = None
        self.func_evals = 0

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        # Encourage periodicity in initial population
        for i in range(self.population_size):
            self.population[i] = self.add_periodicity_bias(self.population[i], lb, ub)

    def add_periodicity_bias(self, individual, lb, ub):
        period = (ub - lb) / self.dim  # Correct periodic calculation
        for i in range(0, self.dim, 2):
            individual[i:i+2] = lb + (i % 2) * period
        return np.clip(individual, lb, ub)

    def differential_evolution(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        while self.func_evals < self.budget:
            for i in range(self.population_size):
                if self.func_evals >= self.budget:
                    break
                # Mutation
                candidates = [index for index in range(self.population_size) if index != i]
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                trial = np.copy(self.population[i])
                jrand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == jrand:
                        trial[j] = mutant[j]

                # Selection
                trial_fitness = func(trial)
                self.func_evals += 1
                if trial_fitness > func(self.population[i]):
                    self.population[i] = trial

    def local_refinement(self, func):
        best_idx = np.argmax([func(ind) for ind in self.population])
        best_solution = self.population[best_idx]
        
        result = minimize(lambda x: -func(x), best_solution, bounds=list(zip(func.bounds.lb, func.bounds.ub)),
                          method='L-BFGS-B', options={'maxfun': self.budget - self.func_evals})
        self.func_evals += result.nfev
        return result.x

    def __call__(self, func):
        self.initialize_population(func.bounds)
        self.differential_evolution(func)
        best_solution = self.local_refinement(func)
        return best_solution
import numpy as np

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.population = None
        self.best_solution = None
        self.best_fitness = float('-inf')
    
    def initialize_population(self, lb, ub):
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        pop_opp = lb + ub - pop  # Quasi-Oppositional Initialization
        self.population = np.vstack((pop, pop_opp))
    
    def differential_evolution_step(self, func, lb, ub):
        for i in range(self.population.shape[0]):
            if self.budget <= 0:
                break
            idxs = [idx for idx in range(self.population.shape[0]) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + 0.8 * (b - c), lb, ub)
            cross_points = np.random.rand(self.dim) < 0.9
            trial = np.where(cross_points, mutant, self.population[i])
            trial_fitness = func(trial)
            self.budget -= 1
            if trial_fitness > func(self.population[i]):
                self.population[i] = trial
                if trial_fitness > self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial
    
    def local_search(self, func, solution, lb, ub):
        # Simple local search encouraging periodicity
        periodic_solution = np.tile(np.mean(solution.reshape(-1, 2), axis=1), int(self.dim / 2))
        fitness = func(periodic_solution)
        self.budget -= 1
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = periodic_solution

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        
        while self.budget > 0:
            self.differential_evolution_step(func, lb, ub)
            for i in range(min(5, self.population.shape[0])):
                self.local_search(func, self.population[i], lb, ub)
        
        return self.best_solution
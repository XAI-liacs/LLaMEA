import numpy as np

class CooperativeHybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.best_solution = None
        self.best_obj = float('inf')
    
    def initialize_population(self, bounds):
        self.lb, self.ub = bounds.lb, bounds.ub
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        self.population_obj = np.array([float('inf')] * self.population_size)
    
    def evaluate_population(self, func):
        objectives = [func(ind) for ind in self.population]
        return np.array(objectives)
    
    def select_parents(self):
        indices = np.random.choice(self.population_size, 3, replace=False)
        return self.population[indices]
    
    def differential_evolution_step(self, target_idx):
        a, b, c = self.select_parents()
        mutant = np.clip(a + self.mutation_factor * (b - c), self.lb, self.ub)
        trial = np.copy(self.population[target_idx])
        crossover = np.random.rand(self.dim) < self.crossover_rate
        trial[crossover] = mutant[crossover]
        return trial
    
    def update_population(self, func):
        new_population = []
        for i in range(self.population_size):
            trial = self.differential_evolution_step(i)
            trial_obj = func(trial)
            if trial_obj < self.population_obj[i]:
                new_population.append(trial)
                self.population_obj[i] = trial_obj
            else:
                new_population.append(self.population[i])
        self.population = np.array(new_population)

    def genetic_algorithm_step(self):
        new_population = []
        for _ in range(self.population_size):
            parents = self.select_parents()
            child = np.mean(parents[:2], axis=0) + np.random.randn(self.dim) * 0.05
            child = np.clip(child, self.lb, self.ub)
            new_population.append(child)
        self.population = np.array(new_population)
    
    def diversity_adaptive(self):
        diversity = np.std(self.population, axis=0).mean()
        self.crossover_rate = np.clip(0.5 + 0.3 * (1 - diversity), 0.1, 0.9)
        self.mutation_factor = np.clip(0.5 + 0.3 * diversity, 0.1, 1.0)
    
    def __call__(self, func):
        self.initialize_population(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            self.population_obj = self.evaluate_population(func)
            evaluations += self.population_size
            
            best_idx = np.argmin(self.population_obj)
            if self.population_obj[best_idx] < self.best_obj:
                self.best_obj = self.population_obj[best_idx]
                self.best_solution = self.population[best_idx]
            
            if evaluations % (2 * self.population_size) < self.population_size:
                self.update_population(func)
            else:
                self.genetic_algorithm_step()
            
            self.diversity_adaptive()
        
        return self.best_solution
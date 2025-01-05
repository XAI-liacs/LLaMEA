import numpy as np

class AdaptiveGenomeRecombination:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.population = None
        self.scores = None
        self.best_solution = None
        self.best_score = np.inf
        self.mutation_rate = 0.1
    
    def _initialize_population(self, lb, ub):
        self.population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.scores = np.full(self.population_size, np.inf)
    
    def _evaluate(self, func):
        for i in range(self.population_size):
            score = func(self.population[i])
            if score < self.scores[i]:
                self.scores[i] = score
            if score < self.best_score:
                self.best_score = score
                self.best_solution = self.population[i]
    
    def _select_parents(self):
        idx = np.argsort(self.scores)
        return self.population[idx[:self.population_size // 2]]
    
    def _recombine(self, parents, lb, ub):
        offspring = np.empty((self.population_size // 2, self.dim))
        for i in range(self.population_size // 2):
            p1, p2 = parents[np.random.choice(parents.shape[0], 2, replace=False)]
            crossover_point = np.random.randint(1, self.dim)
            offspring[i, :crossover_point] = p1[:crossover_point]
            offspring[i, crossover_point:] = p2[crossover_point:]
            if np.random.rand() < self.mutation_rate:
                mutation = np.random.normal(0, 0.1, self.dim) * (ub - lb)
                offspring[i] += mutation
        offspring = np.clip(offspring, lb, ub)
        return offspring
    
    def _update_mutation_rate(self):
        self.mutation_rate = np.random.uniform(0.05, 0.2)
    
    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)
        
        eval_count = 0
        while eval_count < self.budget:
            self._evaluate(func)
            eval_count += self.population_size
            
            parents = self._select_parents()
            offspring = self._recombine(parents, self.lb, self.ub)
            
            if eval_count < self.budget:
                self.population[:self.population_size // 2] = parents
                self.population[self.population_size // 2:] = offspring
            
            self._update_mutation_rate()
        
        return self.best_solution, self.best_score
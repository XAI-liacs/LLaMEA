import numpy as np

class Quantum_DE:
    def __init__(self, budget, dim, population_size=20, F=0.5, CR=0.9, quantum_prob=0.3):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.quantum_prob = quantum_prob
        self.evaluations = 0
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        best_global_position = None
        best_global_value = float('inf')
        
        while self.evaluations < self.budget:
            new_population = []
            for i in range(self.population_size):
                if np.random.rand() < self.quantum_prob:
                    mutant = self.quantum_mutation(population[i], lb, ub)
                else:
                    mutant = self.mutation(population, i, lb, ub)
                
                trial = self.crossover(population[i], mutant, lb, ub)
                value = func(trial)
                self.evaluations += 1
                
                if value < best_global_value:
                    best_global_value = value
                    best_global_position = trial
                
                if value < func(population[i]):
                    new_population.append(trial)
                else:
                    new_population.append(population[i])
                
                if self.evaluations >= self.budget:
                    break
            
            population = new_population
        
        return best_global_position

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def mutation(self, population, index, lb, ub):
        indices = list(range(self.population_size))
        indices.remove(index)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = population[a] + self.F * (population[b] - population[c])
        return np.clip(mutant, lb, ub)
    
    def crossover(self, target, mutant, lb, ub):
        trial = np.copy(target)
        for j in range(self.dim):
            if np.random.rand() < self.CR:
                trial[j] = mutant[j]
        return np.clip(trial, lb, ub)
    
    def quantum_mutation(self, position, lb, ub):
        step_size = np.random.rand(self.dim) * (ub - lb) * 0.1
        q_position = position + step_size * np.where(np.random.rand(self.dim) < 0.5, 1, -1)
        return np.clip(q_position, lb, ub)
import numpy as np

class QuantumEnhancedDEWithAdaptiveQuantumWalks:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.positions = None
        self.best_position = None
        self.best_score = float('inf')
        self.initial_mutation_factor = 0.5
    
    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def quantum_walk(self, individual):
        # Apply a quantum-inspired walk to explore new areas
        step_size = np.random.normal(0, 1, self.dim)
        walk = np.sign(np.random.rand(self.dim) - 0.5) * step_size
        new_position = individual + self.mutation_factor * walk
        return new_position

    def adaptive_mutation_factor(self, evaluations):
        # Adjust mutation factor based on evaluation progress
        return self.initial_mutation_factor * (1 - evaluations / self.budget)
    
    def differential_evolution(self, func, target_idx):
        lb, ub = func.bounds.lb, func.bounds.ub
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        
        mutant_vector = self.positions[a] + self.mutation_factor * (self.positions[b] - self.positions[c])
        trial_vector = np.copy(self.positions[target_idx])
        
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        trial_vector[crossover_mask] = mutant_vector[crossover_mask]
        
        trial_vector = np.clip(trial_vector, lb, ub)
        return trial_vector, func(trial_vector)
    
    def __call__(self, func):
        self.initialize_positions(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                trial_vector, trial_score = self.differential_evolution(func, i)
                evaluations += 1
                
                if trial_score < self.best_score:
                    self.best_score = trial_score
                    self.best_position = trial_vector
                
                if trial_score < func(self.positions[i]):
                    self.positions[i] = trial_vector
                
                self.mutation_factor = self.adaptive_mutation_factor(evaluations)
        
        # Enhanced exploration with quantum walks
        for i in range(self.population_size):
            if evaluations >= self.budget:
                break
            
            quantum_position = self.quantum_walk(self.positions[i])
            quantum_position = np.clip(quantum_position, func.bounds.lb, func.bounds.ub)
            quantum_score = func(quantum_position)
            evaluations += 1
            
            if quantum_score < self.best_score:
                self.best_score = quantum_score
                self.best_position = quantum_position
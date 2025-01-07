import numpy as np

class QuantumInspiredDESelfAdaptiveQBC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.initial_mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.positions = None
        self.best_position = None
        self.best_score = float('inf')
        self.mutation_factor = self.initial_mutation_factor
    
    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def quantum_bit_collapse(self, individual):
        # Dynamic quantum bit collapse for refined exploration
        quantum_bit = np.random.rand(self.dim) < (self.mutation_factor * 0.5 + 0.5)
        quantum_flip = np.where(quantum_bit, 1, -1)
        collapsed_individual = individual + self.mutation_factor * quantum_flip
        return collapsed_individual
    
    def differential_evolution(self, func, target_idx, evaluations):
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
                
                trial_vector, trial_score = self.differential_evolution(func, i, evaluations)
                evaluations += 1
                
                if trial_score < self.best_score:
                    self.best_score = trial_score
                    self.best_position = trial_vector
                
                if trial_score < func(self.positions[i]):
                    self.positions[i] = trial_vector
                
                # Self-adapting mutation factor based on convergence
                convergence_factor = np.tanh(evaluations / self.budget)
                self.mutation_factor = self.initial_mutation_factor * (1 - convergence_factor)
        
        # Quantum-inspired search with dynamic collapse
        for i in range(self.population_size):
            if evaluations >= self.budget:
                break
            
            collapsed_position = self.quantum_bit_collapse(self.positions[i])
            collapsed_position = np.clip(collapsed_position, func.bounds.lb, func.bounds.ub)
            collapsed_score = func(collapsed_position)
            evaluations += 1
            
            if collapsed_score < self.best_score:
                self.best_score = collapsed_score
                self.best_position = collapsed_position
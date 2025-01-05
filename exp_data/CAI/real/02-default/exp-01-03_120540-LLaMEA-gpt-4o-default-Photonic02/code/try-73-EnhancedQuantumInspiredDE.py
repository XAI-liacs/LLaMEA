import numpy as np

class EnhancedQuantumInspiredDE:
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
        self.mutation_rates = np.full(self.population_size, 0.5)
        self.crossover_rates = np.full(self.population_size, 0.7)
    
    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def quantum_inspired_mutation(self, individual, mutation_rate):
        quantum_bit = np.random.rand(self.dim) < 0.5
        quantum_flip = np.where(quantum_bit, 1, -1)
        mutated_individual = individual + mutation_rate * quantum_flip
        return mutated_individual
    
    def differential_evolution(self, func, target_idx, evaluations):
        lb, ub = func.bounds.lb, func.bounds.ub
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        
        mutant_vector = self.positions[a] + self.mutation_rates[target_idx] * (self.positions[b] - self.positions[c])
        trial_vector = np.copy(self.positions[target_idx])
        
        crossover_mask = np.random.rand(self.dim) < self.crossover_rates[target_idx]
        trial_vector[crossover_mask] = mutant_vector[crossover_mask]
        
        trial_vector = np.clip(trial_vector, lb, ub)
        return trial_vector, func(trial_vector)
    
    def self_adaptive_control(self, trial_score, current_score, idx):
        if trial_score < current_score:
            self.mutation_rates[idx] = min(1.0, self.mutation_rates[idx] * 1.1)
            self.crossover_rates[idx] = min(1.0, self.crossover_rates[idx] * 1.1)
        else:
            self.mutation_rates[idx] = max(0.1, self.mutation_rates[idx] * 0.9)
            self.crossover_rates[idx] = max(0.1, self.crossover_rates[idx] * 0.9)

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
                
                # Self-adaptive parameter control
                self.self_adaptive_control(trial_score, func(self.positions[i]), i)
        
        # Quantum-inspired global search for final optimization
        for i in range(self.population_size):
            if evaluations >= self.budget:
                break
            
            mutated_position = self.quantum_inspired_mutation(self.positions[i], self.mutation_rates[i])
            mutated_position = np.clip(mutated_position, func.bounds.lb, func.bounds.ub)
            mutated_score = func(mutated_position)
            evaluations += 1
            
            if mutated_score < self.best_score:
                self.best_score = mutated_score
                self.best_position = mutated_position
import numpy as np

class QuantumInspiredBiogeographyOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.immigration_rate = 0.7
        self.emigration_rate = 0.3
        self.positions = None
        self.best_position = None
        self.best_score = float('inf')
    
    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def habitat_migration(self, habitat, other_habitat):
        migration_mask = np.random.rand(self.dim) < self.immigration_rate
        new_habitat = np.copy(habitat)
        new_habitat[migration_mask] = other_habitat[migration_mask]
        return new_habitat
    
    def quantum_inspired_mutation(self, habitat):
        quantum_bit = np.random.rand(self.dim) < 0.5
        quantum_flip = np.where(quantum_bit, 1, -1)
        mutated_habitat = habitat + self.emigration_rate * quantum_flip
        return mutated_habitat
    
    def __call__(self, func):
        self.initialize_positions(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                # Select two random habitats for migration
                indices = list(range(self.population_size))
                indices.remove(i)
                other_habitat_idx = np.random.choice(indices)
                
                # Perform habitat migration
                trial_habitat = self.habitat_migration(self.positions[i], self.positions[other_habitat_idx])
                trial_habitat = np.clip(trial_habitat, func.bounds.lb, func.bounds.ub)
                trial_score = func(trial_habitat)
                evaluations += 1
                
                if trial_score < self.best_score:
                    self.best_score = trial_score
                    self.best_position = trial_habitat
                
                if trial_score < func(self.positions[i]):
                    self.positions[i] = trial_habitat
            
            # Quantum-inspired mutation for enhanced exploration
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                mutated_habitat = self.quantum_inspired_mutation(self.positions[i])
                mutated_habitat = np.clip(mutated_habitat, func.bounds.lb, func.bounds.ub)
                mutated_score = func(mutated_habitat)
                evaluations += 1
                
                if mutated_score < self.best_score:
                    self.best_score = mutated_score
                    self.best_position = mutated_habitat
import numpy as np

class EnhancedQuantumInspiredDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.initial_mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.positions = None
        self.best_position = None
        self.best_score = float('inf')
    
    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def dynamic_quantum_rotation(self, individual, generation_ratio):
        # Apply dynamic quantum rotation to enhance exploration
        rotation_angle = np.pi * generation_ratio * (np.random.rand(self.dim) - 0.5)
        rotated_individual = individual * np.cos(rotation_angle) + np.sin(rotation_angle)
        return rotated_individual
    
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
            generation_ratio = evaluations / self.budget
            self.mutation_factor = self.initial_mutation_factor * (1 - generation_ratio)
            
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
        
        # Dynamic quantum rotation for final optimization
        for i in range(self.population_size):
            if evaluations >= self.budget:
                break
            
            dynamic_rotated_position = self.dynamic_quantum_rotation(self.positions[i], generation_ratio)
            dynamic_rotated_position = np.clip(dynamic_rotated_position, func.bounds.lb, func.bounds.ub)
            rotated_score = func(dynamic_rotated_position)
            evaluations += 1
            
            if rotated_score < self.best_score:
                self.best_score = rotated_score
                self.best_position = dynamic_rotated_position